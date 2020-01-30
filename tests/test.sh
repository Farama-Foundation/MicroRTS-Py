# set -e
for f in *.py; do
    echo -e "running $f \n" 
    python "$f"; 
    echo -e "\n\n"
    echo -e "$f finished running"
    echo "========================================"
done