# Convert & resize images to 128x128, jpg
for file in $1*
do
	echo $file
	file_name=$(echo $file | sed -r "s/.+\/(.+)\..+/\1/")
	convert $file -resize 128x128 $1$file_name.jpg
done

# Remove .png files
rm $1*.png

# Rename images minus angle to plus angle
python3 rename.py $1
