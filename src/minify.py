from python_minifier import minify

with open('driver.py') as fin:
    code = str(fin.read())

print (minify(code, rename_globals=False, rename_locals=True))

