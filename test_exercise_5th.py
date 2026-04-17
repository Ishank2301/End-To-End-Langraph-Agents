from collections import deque
from typing import Callable, Dict, Any, Deque, Tuple
import ast, operator, math, io, sys, traceback, textwrap

# A compact, general-purpose "AI agent" skeleton for experimentation in a Jupyter notebook.
# GitHub Copilot-style helper: modular components (tools, memory, planner, executor).
# Self-contained: no external LLM; behaviors are heuristic and deterministic.

# --- Utilities ---
SAFE_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
SAFE_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg
}

def safe_eval_expr(expr: str) -> Any:
    """
    Safely evaluate arithmetic/math expressions using AST (no arbitrary exec).
    Supports math.* names (sin, cos, factorial via math) and numeric literals.
    """
    expr = expr.strip()
    if not expr:
        raise ValueError("Empty expression")
    node = ast.parse(expr, mode="eval").body

    def _eval(n):
        if isinstance(n, ast.Num):
            return n.n
        if isinstance(n, ast.Constant):  # Python 3.8+
            return n.value
        if isinstance(n, ast.BinOp):
            op = SAFE_OPERATORS[type(n.op)]
            return op(_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            op = SAFE_OPERATORS[type(n.op)]
            return op(_eval(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            fname = n.func.id
            if fname in SAFE_NAMES:
                args = [_eval(a) for a in n.args]
                return SAFE_NAMES[fname](*args)
        if isinstance(n, ast.Name):
            if n.id in SAFE_NAMES:
                return SAFE_NAMES[n.id]
        raise ValueError(f"Unsupported expression: {ast.dump(n)}")
    return _eval(node)

# --- Tool abstraction ---
class Tool:
    def __init__(self, name: str, func: Callable[..., Any], description: str = ""):
        self.name = name
        self.func = func
        self.description = description

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)

# --- Agent ---
class Agent:
    def __init__(self, name: str = "Agent", memory_size: int = 50):
        self.name = name
        self.memory: Deque[str] = deque(maxlen=memory_size)
        self.tools: Dict[str, Tool] = {}
        self.verbose = False

    def add_tool(self, tool: Tool):
        self.tools[tool.name.lower()] = tool

    def remember(self, note: str):
        self.memory.appendleft(note)

    def recall(self, n: int = 5) -> Tuple[str, ...]:
        return tuple(list(self.memory)[:n])

    def plan(self, instruction: str) -> Tuple[str, str]:
        """
        Very simple planner: route instruction to a tool based on keywords,
        returns (tool_name, cleaned_instruction).
        """
        s = instruction.lower()
        if any(k in s for k in ("calculate", "compute", "evaluate", "math", "solve")):
            return "calculator", instruction
        if any(k in s for k in ("run python", "execute python", "python:", "write python", "code")):
            return "python", instruction
        if any(k in s for k in ("search", "find", "look up")):
            return "search", instruction
        if any(k in s for k in ("summarize", "summarise", "shorten", "brief")):
            return "summarizer", instruction
        # default to a "thinker" that just echoes / records
        return "think", instruction

    def act(self, instruction: str) -> Any:
        tool_name, cleaned = self.plan(instruction)
        tool = self.tools.get(tool_name)
        if not tool:
            self.remember(f"Unable to handle: {instruction}")
            return f"No tool named '{tool_name}' registered."
        if self.verbose:
            print(f"[{self.name}] Using tool: {tool.name} for instruction: {instruction}")
        try:
            result = tool.run(cleaned)
            self.remember(f"Instruction: {instruction} -> Result: {str(result)[:400]}")
            return result
        except Exception as e:
            tb = traceback.format_exc()
            self.remember(f"Instruction: {instruction} -> Error: {e}")
            return f"Error during tool execution: {e}\n{tb}"

    def run(self, instruction: str) -> Any:
        """Single-step run. Could be extended to multi-step loops."""
        return self.act(instruction)

# --- Built-in tools ---
def calculator_tool(text: str) -> str:
    # Extract the expression heuristically
    # e.g., "Calculate 2 + 2" -> "2 + 2"
    expr = text
    for kw in ("calculate", "compute", "evaluate", "math:", "math"):
        expr = expr.replace(kw, "")
    expr = expr.strip(" :")
    try:
        val = safe_eval_expr(expr)
        return f"{val}"
    except Exception as e:
        return f"Calculator error: {e}"

def summarizer_tool(text: str) -> str:
    # Extremely simple summarizer: return first 2 sentences or first 200 chars
    body = text.strip()
    if not body:
        return ""
    sentences = body.split(".")
    if len(sentences) > 2:
        return (sentences[0] + "." + sentences[1] + ".").strip()
    return body[:200].strip()

def search_mock_tool(text: str) -> str:
    # Placeholder for real search; returns structured mock response
    query = text.replace("search", "").strip()
    return f"Search results for '{query}': [MockResult1, MockResult2]"

def python_executor_tool(text: str) -> str:
    """
    Execute Python code safely in a restricted namespace and capture stdout.
    Heuristic: extract code following keywords 'run python', 'python:', 'execute python'.
    """
    code = text
    for kw in ("run python", "execute python", "python:", "write python", "code:"):
        code = code.replace(kw, "")
    code = textwrap.dedent(code).strip()
    if not code:
        return "No code provided."
    # Allowed builtins (very restricted)
    safe_builtins = {"print": print, "range": range, "len": len, "min": min, "max": max, "sum": sum}
    safe_globals = {"__builtins__": safe_builtins, "math": math}
    safe_locals = {}
    old_stdout = sys.stdout
    buf = io.StringIO()
    try:
        sys.stdout = buf
        # Try to evaluate as expression first
        try:
            result = eval(code, safe_globals, safe_locals)
            if result is not None:
                print(repr(result))
        except SyntaxError:
            exec(code, safe_globals, safe_locals)
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout = old_stdout
    return buf.getvalue().strip()

def thinker_tool(text: str) -> str:
    # Default fallback: echo + small heuristics
    txt = text.strip()
    if len(txt) > 300:
        return txt[:300] + "..."
    return txt

# --- Assemble an agent with tools ---
agent = Agent(name="SmartAgent", memory_size=100)
agent.add_tool(Tool("calculator", calculator_tool, "Evaluate math expressions safely"))
agent.add_tool(Tool("summarizer", summarizer_tool, "Summarize text simply"))
agent.add_tool(Tool("search", search_mock_tool, "Mock web search"))
agent.add_tool(Tool("python", python_executor_tool, "Execute Python code in a restricted environment"))
agent.add_tool(Tool("think", thinker_tool, "Fallback echo/think tool"))

# Example usage:
if __name__ == "__main__":
    agent.verbose = False
    print(agent.run("Calculate 12 * (5 + 7)"))
    print(agent.run("Summarize: Artificial intelligence is the simulation of human intelligence processes by machines. It includes learning, reasoning, and self-correction."))
    print(agent.run("python: for i in range(3): print('line', i)"))
    # Inspect memory (recent actions)
    print("Memory:", list(agent.recall(5)))
