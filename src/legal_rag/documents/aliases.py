def build_law_aliases(law_name: str) -> list[str]:
    aliases = [law_name]
    if law_name.startswith("中华人民共和国"):
        short_name = law_name.removeprefix("中华人民共和国")
        if short_name and short_name not in aliases:
            aliases.append(short_name)
    return aliases

