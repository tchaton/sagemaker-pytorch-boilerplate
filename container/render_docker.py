from jinja2 import Template

CMD = "RUN pip install "

requirements = []
with open("requirements.txt") as file_:
    for l in file_:
        if ';' in l:
            requirements.append(CMD + l.split(';')[0] + '\n')
        else:
            requirements.append(CMD + l)
with open("docker_template.jinja2") as file_:
    template = Template(file_.read())
rendered = template.render(requirements=''.join(requirements))
with open("Dockerfile", "w") as file_:
    file_.write(rendered)