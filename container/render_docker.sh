requirements=$(cat requirements.txt | python3 -c "import sys, json; print(''.join(['RUN pip install '+l for l in sys.stdin]))")
docker_template=$(cat docker_template.txt) 
output=$(sed -i -e "s/{{requirements}}/$requirements/g" $docker_template)
echo $output