# !/bin/bash

echo $(echo "Hello $(whoami)")

docker exec -ti mindspore /bin/bash