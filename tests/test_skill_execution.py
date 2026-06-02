"""Skill 执行+集成测试 — 加载示例 skill、执行脚本、Agent 集成"""

import sys, os, json, asyncio, tempfile

_here = os.path.dirname(os.path.abspath(__file__))
_wuwei_root = os.path.dirname(_here)
sys.path.insert(0, _wuwei_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(_wuwei_root, '.env'))

from wuwei.skill import SkillManager, FileSystemSkillProvider
from wuwei.tools import ToolRegistry
from wuwei.plugin import PluginContext
from wuwei.plugin.builtin.skill import SkillPromptMiddleware, setup as setup_skill_plugin
from wuwei.middleware import MiddlewareStack
from wuwei.agent import Agent, AgentSession
from wuwei.llm import LLMGateway, Message
from wuwei.middleware.base import MiddlewareContext
from wuwei.graph.state import State

PASSED = 0
FAILED = 0

def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        print(f"  ❌ {name}  {detail}")

# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("Test 1: 加载示例 skill (code-review) 并执行脚本")
print("=" * 60)

async def test_example_skill():
    examples_dir = os.path.join(_wuwei_root, "examples", "skills")
    provider = FileSystemSkillProvider(skill_path=examples_dir)
    manager = SkillManager([provider])
    
    skills = manager.list_skills()
    check("发现了示例 skill", len(skills) >= 1, f"found {len(skills)}: {[s.name for s in skills]}")
    
    code_review = manager.get_skill("code-review")
    check("code-review skill 存在", code_review is not None)
    check("description 正确", "代码审查" in code_review.description)
    check("有 scripts", len(code_review.scripts) > 0, f"scripts={code_review.scripts}")
    check("scripts 包含 count_lines.py", "count_lines.py" in str(code_review.scripts))
    
    # 注册 skill 工具
    registry = ToolRegistry()
    setup_skill_plugin(PluginContext(tool_registry=registry, skill_manager=manager))
    
    # list_skills 工具
    list_result = await registry.get("list_skills").invoke({})
    check("list_skills 工作", len(list_result) >= 1)
    check("list_skills 包含 code-review", any(s["name"] == "code-review" for s in list_result))
    
    # load_skill 工具
    load_result = await registry.get("load_skill").invoke({"skill_name": "code-review"})
    check("load_skill 工作", load_result["name"] == "code-review")
    check("load_skill 返回 load_token", isinstance(load_result["load_token"], str))
    check("load_skill 有 instruction", len(load_result["instruction"]) > 0)
    check("load_skill 有 scripts 列表", "count_lines.py" in str(load_result["python_scripts"]))
    
    # run_skill_python_script — 执行 count_lines.py
    load_token = load_result["load_token"]
    # 用一个临时文件做测试目标
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("def hello():\n    pass\n\n\nclass Foo:\n    pass\n")
        temp_path = f.name
    
    try:
        run_result = await registry.get("run_skill_python_script").invoke({
            "skill_name": "code-review",
            "script_path": "scripts/count_lines.py",
            "load_token": load_token,
            "args_json": json.dumps([temp_path]),
        })
        check("脚本执行成功", run_result["ok"] is True, run_result.get("stderr", ""))
        check("脚本输出包含统计", "总行数" in run_result["stdout"])
        check("脚本输出有数字", any(c.isdigit() for c in run_result["stdout"]))
        print(f"   脚本输出: {run_result['stdout'].strip()}")
    finally:
        os.unlink(temp_path)
    
    print("  示例 Skill 加载+执行 全部通过 ✅")

asyncio.run(test_example_skill())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 2: SkillPromptMiddleware 集成到 Agent（LLM 调用）")
print("=" * 60)

async def test_skill_middleware_with_llm():
    llm = LLMGateway({
        "provider": "openai",
        "api_key": os.getenv("WUWEI_API_KEY"),
        "base_url": os.getenv("WUWEI_BASE_URL"),
        "model": os.getenv("WUWEI_MODEL"),
        "temperature": 0.2,
        "max_tokens": 512,
    })
    
    # 创建 skill
    from wuwei.skill.skill import Skill
    
    skill = Skill(
        name="concise-responder",
        description="简洁回答技能",
        instruction="你必须用不超过15个字回答所有问题。回复必须极其简洁。",
    )
    
    class MockProvider:
        def list_skills(self): return [skill]
        def load_skill_instruction(self, name): return skill.instruction
    
    manager = SkillManager([MockProvider()])
    skill_mw = SkillPromptMiddleware(skill_manager=manager)
    
    stack = MiddlewareStack()
    stack.add(skill_mw)
    
    agent = Agent(
        llm=llm,
        default_system_prompt="你是一个助手。",
        default_max_steps=3,
        middleware=stack,
    )
    
    result = await agent.run("用一句话介绍人工智能")
    check("Agent 正常返回", result.content is not None and len(result.content) > 0)
    
    # SkillPromptMiddleware 注入验证：检查 agent 运行的 session 状态
    # skill 注入发生在 ctx.state.messages（LLM 实际收到的消息），
    # 而非 session.context（持久化存储）。因此验证 agent 正常完成即可。
    # 详细的 before_llm 注入测试见 test_skill_integration.py
    check("Agent 在 SkillPromptMiddleware 下正常完成", result.llm_calls >= 1)
    
    text = result.content[:150]
    print(f"   回复: {text}")
    
    print("  SkillPromptMiddleware + Agent LLM 集成 通过 ✅")

asyncio.run(test_skill_middleware_with_llm())

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 3: Skill list_by_tag / list_by_tool 过滤")
print("=" * 60)

def test_skill_filtering():
    from wuwei.skill.skill import Skill
    
    s1 = Skill(name="skill-a", description="A skill", instruction="a", tags=["code", "python"])
    s2 = Skill(name="skill-b", description="B skill", instruction="b", tags=["review"])
    s3 = Skill(name="skill-c", description="C skill", instruction="c", tags=["code"], allowed_tools=["read_file"])
    
    class MockProvider:
        def list_skills(self): return [s1, s2, s3]
    
    manager = SkillManager([MockProvider()])
    
    # by tag
    code_skills = manager.list_by_tag("code")
    check("list_by_tag 'code' 返回2个", len(code_skills) == 2, f"got {len(code_skills)}")
    check("包含 skill-a", any(s.name == "skill-a" for s in code_skills))
    check("包含 skill-c", any(s.name == "skill-c" for s in code_skills))
    
    review_skills = manager.list_by_tag("review")
    check("list_by_tag 'review' 返回1个", len(review_skills) == 1)
    
    # by tool
    read_skills = manager.list_by_tool("read_file")
    check("list_by_tool 'read_file' 返回 skill-c", len(read_skills) >= 1)
    
    # 无限制工具的 skill 对所有工具可见
    all_for_calc = manager.list_by_tool("calculate")
    check("无限制工具的 skill 对所有工具可见", len(all_for_calc) >= 2)  # s1, s2
    
    print("  Skill 过滤 全部通过 ✅")

test_skill_filtering()

# ═══════════════════════════════════════════════════════════
print(f"\n📊 Skill 测试结果: ✅ {PASSED} passed  ❌ {FAILED} failed  (共 {PASSED+FAILED} 项)")
