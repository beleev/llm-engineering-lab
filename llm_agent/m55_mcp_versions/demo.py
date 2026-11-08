"""固定旧 stdio 配置与现代 HTTP 报文约定；本地资源服务器授权反例。"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import base64
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from llm_agent.core.mcp import MCPClient

LEGACY='2025-06-18'
MODERN='2026-07-28'
VERSION_KEY='io.modelcontextprotocol/protocolVersion'
MODES=('legacy','modern','unknown','audience','expired','scope','cancel','timeout')

@dataclass(frozen=True)
class Credential:
    audience: str
    expires: int
    scopes: frozenset[str]

# 不接受客户端自报 claims：opaque token 在服务端查固定夹具；不实现完整 OAuth issuer。
TOKENS={'ok':Credential('lab-mcp',200,frozenset({'tools:read'})),
        'audience':Credential('other-api',200,frozenset({'tools:read'})),
        'expired':Credential('lab-mcp',50,frozenset({'tools:read'})),
        'scope':Credential('lab-mcp',200,frozenset())}

def error(id,code,message,data=None):
    return {'jsonrpc':'2.0','id':id,'error':{'code':code,'message':message,**({'data':data} if data else {})}}

def dispatch(request,header_version):
    id=request.get('id');params=request.get('params',{})
    meta=params.get('_meta',{})
    if not isinstance(meta,dict):return error(id,-32602,'Invalid metadata')
    version=meta.get(VERSION_KEY)
    if not version or header_version!=version:
        return error(id,-32020,'HeaderMismatch')
    if version!=MODERN:
        return error(id,-32022,'Unsupported protocol version',{'supported':[MODERN],'requested':version})
    method=request.get('method')
    info={'name':'lab-http-resource','version':'1'}
    meta={'io.modelcontextprotocol/serverInfo':info}
    if method=='server/discover':
        result={'supportedVersions':[MODERN],'capabilities':{'tools':{}}}
    elif method=='tools/list':
        result={'tools':[{'name':'echo','description':'Return text without side effects','inputSchema':{'type':'object','properties':{'text':{'type':'string'}},'required':['text']}}]}
    elif method=='tools/call':
        args=params.get('arguments',{})
        if params.get('name')!='echo' or not isinstance(args.get('text'),str):return error(id,-32602,'Invalid params')
        result={'content':[{'type':'text','text':args['text']}],'isError':False}
    else:return error(id,-32601,'Method not found')
    return {'jsonrpc':'2.0','id':id,'result':{'resultType':'complete',**result,'_meta':meta}}

class Execution:
    """工具内部的合作取消与截止时间。不是 MCP tasks 扩展实现。"""
    def __init__(self):self.cancelled=threading.Event();self.committed=0
    def run(self,deadline,now):
        if self.cancelled.is_set():return 'cancelled'
        if now>=deadline:return 'deadline'
        self.committed+=1;return 'committed'

@contextmanager
def resource_server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,body):
            data=json.dumps(body).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
            if status==401:self.send_header('WWW-Authenticate','Bearer realm="lab-mcp"')
            self.end_headers();self.wfile.write(data)
        def do_POST(self):
            if self.path!='/mcp':return self.reply(404,{'error':'path'})
            if self.headers.get('Origin') is not None:return self.reply(403,{'error':'browser_origin_not_allowed'})
            if self.headers.get('Host')!=f'127.0.0.1:{self.server.server_port}':return self.reply(403,{'error':'host'})
            authorization=self.headers.get('Authorization','')
            if not authorization.startswith('Bearer '):return self.reply(401,{'error':'missing_bearer'})
            token=authorization.removeprefix('Bearer ')
            credential=TOKENS.get(token)
            if credential is None or credential.audience!='lab-mcp' or credential.expires<=100:
                return self.reply(401,{'error':'invalid_token'})
            if 'tools:read' not in credential.scopes:return self.reply(403,{'error':'insufficient_scope'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16384:raise ValueError('body size')
                body=json.loads(self.rfile.read(size))
                if not isinstance(body,dict) or body.get('jsonrpc')!='2.0' or not isinstance(body.get('params',{}),dict):raise ValueError('request')
            except (ValueError,TypeError):return self.reply(400,{'error':'malformed'})
            if 'id' not in body:return self.reply(400,error(None,-32600,'Only requests supported'))
            if self.headers.get('Mcp-Method')!=body.get('method'):
                return self.reply(400,error(body['id'],-32020,'HeaderMismatch: Mcp-Method'))
            if body.get('method')=='tools/call':
                try:name=decode_header(self.headers.get('Mcp-Name',''))
                except (ValueError,UnicodeError):return self.reply(400,error(body['id'],-32020,'HeaderMismatch: Mcp-Name'))
                if name!=body['params'].get('name'):return self.reply(400,error(body['id'],-32020,'HeaderMismatch: Mcp-Name'))
            result=dispatch(body,self.headers.get('MCP-Protocol-Version'))
            code=result.get('error',{}).get('code')
            self.reply(404 if code==-32601 else 400 if code is not None else 200,result)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}/mcp'
    finally:server.shutdown();server.server_close();thread.join(timeout=2)

def encode_header(value):
    if not value.isascii() or value.strip()!=value or any(ord(c)<32 for c in value) or (value.startswith('=?base64?') and value.endswith('?=')):
        return '=?base64?'+base64.b64encode(value.encode()).decode()+'?='
    return value

def decode_header(value):
    if value.startswith('=?base64?') and value.endswith('?='):
        return base64.b64decode(value[9:-2],validate=True).decode()
    return value

def call(url,version=MODERN,token='ok',method='tools/list',arguments=None,headers_override=None):
    params={'_meta':{VERSION_KEY:version,'io.modelcontextprotocol/clientCapabilities':{},
                     'io.modelcontextprotocol/clientInfo':{'name':'lab','version':'1'}}}
    if arguments is not None:params.update(arguments)
    headers={'Authorization':'Bearer '+token,'Content-Type':'application/json',
             'Accept':'application/json, text/event-stream','MCP-Protocol-Version':version,'Mcp-Method':method}
    if method=='tools/call':headers['Mcp-Name']=encode_header(params.get('name',''))
    headers.update(headers_override or {})
    req=Request(url,data=json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params}).encode(),
                headers=headers)
    try:
        with urlopen(req,timeout=2) as response:return response.status,json.load(response)
    except HTTPError as exc:
        with exc:return exc.code,json.load(exc)

def experiment(mode='modern'):
    if mode not in MODES:raise ValueError(mode)
    status=200;rejected=False;commits=0
    if mode=='legacy':
        server=Path(__file__).resolve().parents[1]/'m09_mcp/server.py'
        with MCPClient('legacy',[sys.executable,str(server)]) as client:
            assert client.server_info['protocolVersion']==LEGACY
            count=len(client.list_tools())
        steps=['真实子进程 stdio：initialize → initialized → tools/list。',f'固定旧协议 {LEGACY}，发现 {count} 个工具。']
    else:
        with resource_server() as url:
            status,response=call(url,'1900-01-01' if mode=='unknown' else MODERN,
                                 mode if mode in ('audience','expired','scope') else 'ok')
            rejected=status!=200 or 'error' in response
            if mode=='modern':
                _,discovery=call(url,method='server/discover');assert 'result' in discovery
                _,echo=call(url,method='tools/call',arguments={'name':'echo','arguments':{'text':'hello'}})
                assert echo['result']['content'][0]['text']=='hello'
            if mode in ('cancel','timeout'):
                execution=Execution()
                if mode=='cancel':execution.cancelled.set()
                outcome=execution.run(99 if mode=='timeout' else 110,100)
                commits=execution.committed;assert commits==0
                response={'execution':outcome}
            steps=[f'实际请求本机 HTTP 端点；协议版本 {MODERN} 放在请求元数据及 header 中。',
                   json.dumps(response,ensure_ascii=False,sort_keys=True)]
        if mode in ('audience','expired'):assert status==401
        if mode=='scope':assert status==403
        if mode=='unknown':assert status==400 and response['error']['code']==-32022
    return {'metrics':{'HTTP 状态':str(status) if mode!='legacy' else '不适用','协议或授权拒绝':rejected,'取消/超时后提交':commits},
            'steps':steps,'note':'HTTP 资源端真实运行于 loopback，固定时钟和 opaque 凭据用于授权反例；未实现 OAuth 登录、令牌签发、SSE 或完整互操作认证。取消/超时检验工具执行层的合作停止，不等于网络断连自动撤销副作用。stdio 的环境凭据与 HTTP 授权流程分开。'}

def main():
    print(json.dumps({m:experiment(m) for m in MODES},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
