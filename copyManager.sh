#!/bin/bash

COUNTER=0
while [ $COUNTER -lt 1 ]; do
    if [ -a ./copyTos3 ]
        then
            sudo chmod +x ./copyTos3;
            sudo ./copyTos3;
            sudo rm ./copyTos3;
    fi
    sleep 1m;
done

