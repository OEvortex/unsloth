import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _get_method(path: Path, class_name: str, func_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == func_name:
                    return child
    raise AssertionError(f"{func_name} not found in {class_name}")


def _arguments_dict_has_keyword(node: ast.AST, keyword: str) -> bool:
    return (
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "arguments" for t in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "dict"
        and any(isinstance(kw, ast.keyword) and kw.arg == keyword for kw in node.value.keywords)
    )


def test_llama_get_peft_model_accepts_target_parameters():
    method = _get_method(
        REPO_ROOT / "unsloth" / "models" / "llama.py",
        "FastLlamaModel",
        "get_peft_model",
    )
    arg_names = [arg.arg for arg in method.args.args]
    assert "target_parameters" in arg_names


def test_llama_get_peft_model_forwards_target_parameters():
    method = _get_method(
        REPO_ROOT / "unsloth" / "models" / "llama.py",
        "FastLlamaModel",
        "get_peft_model",
    )
    assert any(
        _arguments_dict_has_keyword(node, "target_parameters") for node in ast.walk(method)
    )


def test_vision_get_peft_model_accepts_target_parameters():
    method = _get_method(
        REPO_ROOT / "unsloth" / "models" / "vision.py",
        "FastBaseModel",
        "get_peft_model",
    )
    arg_names = [arg.arg for arg in method.args.args]
    assert "target_parameters" in arg_names
