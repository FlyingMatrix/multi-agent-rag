from pathlib import Path
from string import Template
    

class SkillRegistry:
    def __init__(self, base_path: str = "skills"):
        self.base_path = Path(base_path)
        self.cache = {}

    def load(self, skill_name: str) -> Template:
        if skill_name in self.cache:
            return self.cache[skill_name]

        skill_file = self.base_path / f"{skill_name}.md"
        content = skill_file.read_text(encoding="utf-8")
        template = Template(content)

        self.cache[skill_name] = template
        return template

    def render(self, skill_name: str, **kwargs) -> str:
        template = self.load(skill_name)
        return template.safe_substitute(**kwargs)
    

