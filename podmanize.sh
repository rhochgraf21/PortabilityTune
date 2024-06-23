# update python dependencies
pipreqs . > requirements.txt --force    
# update docker build image
podman build -t python-ptuner .