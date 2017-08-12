#!/bin/bash

# Create folders
rm -R faces_aligned

mkdir faces_aligned
cd faces_aligned

mkdir train
cd train
for j in `seq 01 80`;
do
	nameDir=$(printf "%02d" $j)
	mkdir $nameDir
done
cd ..

mkdir test
cd test
for j in `seq 01 80`;
do
	nameDir=$(printf "%02d" $j)
	mkdir $nameDir
done
cd ..

mkdir valid
cd valid
for j in `seq 01 80`;
do
	nameDir=$(printf "%02d" $j)
	mkdir $nameDir
done
cd ..
cd ..

# Process images
cd faces_data
shopt -s nullglob
for i in $(ls);
do

	# Change test/train/valid
	echo "cd $i"
	cd "$i"

	for j in $(ls);
	do

		echo "cd $j"
		cd "$j"

		for image in *;
		do
			echo "$image"
			python ../../../align_image.py \
			--shape-predictor ../../../face_recognizer/shape_predictor_68_face_landmarks.dat \
			--image "$image" \
			--output "../../../faces_aligned/$i/$j/aligned_$image"

		done

		cd ..

	done

	cd ..
	
done