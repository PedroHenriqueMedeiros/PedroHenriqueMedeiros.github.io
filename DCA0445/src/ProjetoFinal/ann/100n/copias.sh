#!/bin/bash

arquivos=$(ls *.jpg)
count=6

for arquivo in $arquivos; do
	echo "$arquivo";
	convert $arquivo -rotate 90 "$count.jpg"
	count=$[count + 1]
done

for arquivo in $arquivos; do
	echo "$arquivo";
	convert $arquivo -rotate 180 "$count.jpg"
	count=$[count + 1]
done

for arquivo in $arquivos; do
	echo "$arquivo";
	convert $arquivo -rotate 270 "$count.jpg"
	count=$[count + 1]
done
