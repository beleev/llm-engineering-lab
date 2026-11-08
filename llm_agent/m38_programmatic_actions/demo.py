"""代码作为行动载体: 受限表达式、适配器、转义后的 UI 与 SVG。"""
from __future__ import annotations
import ast
import html
import json
import math
import operator
from xml.etree import ElementTree as ET

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculate(source, variables):
    if any(type(v) not in {int, float} or not math.isfinite(v) for v in variables.values()):
        raise ValueError("变量必须是有限数字")
    tree = ast.parse(source, mode="eval")
    if len(list(ast.walk(tree))) > 40:
        raise ValueError("表达式超预算")
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            if not math.isfinite(node.value):
                raise ValueError("常量必须是有限数字")
            return node.value
        if isinstance(node, ast.Name) and node.id in variables:
            return variables[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            try:
                value = OPS[type(node.op)](visit(node.left), visit(node.right))
                if not math.isfinite(value):
                    raise ValueError("计算结果超出有限数字范围")
                return value
            except (ZeroDivisionError, OverflowError) as error:
                raise ValueError("无效算术操作") from error
        raise ValueError("只支持数字、已知变量和四则运算")
    return visit(tree)


def adapt(raw):
    if set(raw) != {"order_id", "amount_cents"} or type(raw["amount_cents"]) is not int:
        raise ValueError("响应 Schema 不匹配")
    return {"id": str(raw["order_id"]), "amount": raw["amount_cents"] / 100}


def make_ui(label, amount):
    return f'<article><h2>{html.escape(label)}</h2><p>{amount:.2f}</p></article>'


def make_svg(label):
    root = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width="240", height="80")
    ET.SubElement(root, "text", x="10", y="40").text = label
    return ET.tostring(root, encoding="unicode")


def experiment(mode="validated"):
    if mode not in {"naive", "validated", "rejected"}:
        raise ValueError(mode)
    source = "total * (1 - percent / 100)" if mode != "naive" else "total * (1 - percent)"
    if mode == "rejected":
        source = "__import__('os').system('echo unsafe')"
    try:
        amount = calculate(source, {"total": 100, "percent": 10})
        allowed = True
    except ValueError:
        amount, allowed = None, False
    output = make_ui('<img src=x onerror="run()">', amount or 0)
    svg = make_svg("<script>not executable</script>")
    ET.fromstring(svg)
    checks = [allowed, amount == 90, "<img" not in output, "<script>" not in svg]
    return {"amount": amount, "allowed": allowed, "checks": checks, "adapter": adapt({"order_id": "1004", "amount_cents": 1999}),
            "trace": [{"title": "表达为程序", "detail": source, "ok": allowed},
                      {"title": "业务验收", "detail": f"期望 90, 实际 {amount}", "ok": amount == 90},
                      {"title": "生成界面与媒体", "detail": "用户字符串被转义; SVG 文本节点不作为脚本执行", "ok": checks[2] and checks[3]}]}


def main():
    rows = {m: experiment(m) for m in ("naive", "validated", "rejected")}
    assert rows["validated"]["amount"] == 90
    assert rows["naive"]["amount"] != 90 and not rows["rejected"]["allowed"]
    assert rows["validated"]["adapter"] == {"id": "1004", "amount": 19.99}
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
