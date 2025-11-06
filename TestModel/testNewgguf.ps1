$blob = "$env:USERPROFILE\.ollama\models\blobs\sha256-108e7ff92b78eefd3db4741885104acba514255c11b617d3c7b197a5f46efe89"
$py = 'from gguf import GGUFReader; import json, pathlib
path = pathlib.Path(r"' + $blob + '")
reader = GGUFReader(path)
kv = {k: reader.get_metadata(k) for k in reader.fields}
path.with_suffix(".kv.json").write_text(json.dumps(kv, indent=2))
path.with_suffix(".tensors.txt").write_text("\n".join(t.name for t in reader.tensors))
print("Wrote", path.with_suffix(".kv.json"), "and", path.with_suffix(".tensors.txt"))'
c:\Users\iosuc\AppData\Local\Programs\Python\Python312\python.exe -c $py