"""跨文件接口变更：候选修改与独立验收进程。"""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
MODES=("partial","complete","test-cheat")


def execute(mode):
    if mode not in MODES:
        raise ValueError(mode)
    with TemporaryDirectory(prefix="lab-repo-") as tmp:
        root=Path(tmp)
        workspace=root/"candidate";workspace.mkdir()
        (workspace/"rates.py").write_text("def rate():\n    return 0.2\n")
        # API 从百分数迁移到比例；只改 provider 会破坏 consumer。
        division=" / 100" if mode=="partial" else ""
        (workspace/"invoice.py").write_text(
            "from rates import rate\n"
            "def total(price, quantity):\n"
            f"    return price * quantity * (1 - rate(){division})\n")
        if mode=="test-cheat":
            (workspace/"invoice.py").write_text("def total(price, quantity):\n    return 80\n")
            (workspace/"test_local.py").write_text("assert True\n")
        # 验收器在候选目录之外，使用另一个进程实际导入候选代码。
        checker=root/"verify.py"
        checker.write_text(
            "import json,sys\n"
            "sys.path.insert(0,sys.argv[1])\n"
            "from invoice import total\n"
            "cases=[(100,1,80),(30,3,72),(0,5,0),(12.5,2,20)]\n"
            "ok=[abs(total(p,q)-expected)<1e-9 for p,q,expected in cases]\n"
            "print(json.dumps({'passed':sum(ok),'total':len(ok)}))\n"
            "sys.exit(0 if all(ok) else 1)\n")
        proc=subprocess.run([sys.executable,"-I",str(checker),str(workspace)],
                            capture_output=True,text=True,timeout=10)
        result=json.loads(proc.stdout)
        return {**result,"exit":proc.returncode}


def experiment(mode="complete"):
    result=execute(mode)
    return {"metrics":{"独立用例通过":result["passed"],"独立用例总数":result["total"],"子进程退出码":result["exit"]},
            "steps":["修改 rates.py 的单位会影响 invoice.py；仓库理解首先需要追踪契约和依赖。",
                     "独立验收器放在候选工作区之外，重新启动 Python 导入真实文件；不是检查补丁文本里有没有关键字。",
                     "test-cheat 把局部测试改成永真并硬编码一个样本，但独立用例仍发现失败。",
                     "代码训练还可用 FIM：prefix/suffix/middle；划分必须按仓库/时间，避免同文件片段泄漏。测试生成覆盖输入边界，代码审查另看安全与需求。"],
            "note":"两个模块的受控 API 迁移，候选补丁由程序给定，不是通用 Coding Agent 或安全沙箱；-I 只隔离 Python 环境，不隔离操作系统权限。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert rows["complete"]["metrics"]["子进程退出码"]==0
    assert rows["partial"]["metrics"]["子进程退出码"]!=0
    assert rows["test-cheat"]["metrics"]["子进程退出码"]!=0
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
