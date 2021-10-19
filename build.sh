cd gym_microrts/microrts


rm -rf build microrts.jar
mkdir build
javac -d "./build" -cp "./lib/*" -sourcepath "./src"  $(find ./src/* | grep .java)
cp -a lib/. build/

# hack to remove the weka dependency in build time
# we don't use weka anyway yet it's a 10 MB package
rm build/weka.jar
rm -rf build/bots

cd build
for i in *.jar; do
    echo "adding dependency $i"
    jar xf $i
done
jar cvf microrts.jar *
mv microrts.jar ../microrts.jar
cd ..
rm -rf build