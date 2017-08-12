cd faces_data

echo "Train Data"
for folder in `seq 01 80`;
do
	nameDir=$(printf "%02d" $folder)

	cd train/$nameDir
	numTrainFiles=$(ls -l | grep -v ^l | wc -l)
	cd ../..

	cd test/$nameDir
	numTestFiles=$(ls -l | grep -v ^l | wc -l)
	cd ../..

	cd valid/$nameDir
	numValidFiles=$(ls -l | grep -v ^l | wc -l)
	cd ../..

	line=$(printf "\"%s\",%d,%d,%d" $nameDir $numTrainFiles $numTestFiles $numValidFiles)
	echo "$line"
done 
cd ..