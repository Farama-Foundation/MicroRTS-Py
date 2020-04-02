VERSION=1.4-cuda10.1-cudnn7-runtime-gym-microrts-0.0.1

docker build -t vwxyzjn/gym-microrts:$VERSION  -t vwxyzjn/gym-microrts:latest -f Dockerfile .
docker build -t vwxyzjn/gym-microrts_shared_memory:$VERSION  -t vwxyzjn/gym-microrts_shared_memory:latest -f sharedmemory.Dockerfile .

docker push vwxyzjn/gym-microrts:latest
docker push vwxyzjn/gym-microrts_shared_memory:latest
docker push vwxyzjn/gym-microrts:$VERSION
docker push vwxyzjn/gym-microrts_shared_memory:$VERSION
