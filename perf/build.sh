rm -rf build perf.jar
mkdir build

javac -d "./build" -sourcepath "./src" $(find ./src/* | grep .java)

cd build

jar cvf perf.jar *

mv perf.jar ../perf.jar
cd ..
rm -rf build