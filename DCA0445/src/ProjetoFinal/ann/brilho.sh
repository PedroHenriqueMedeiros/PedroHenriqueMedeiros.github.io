#!/bin/bash

arquivos=$(ls *.jpg)
count=21

for arquivo in $arquivos; do
	echo "$arquivo";
	convert $arquivo -brightness-contrast -20x10 "$count.jpg"
	count=$[count + 1]
done

for arquivo in $arquivos; do
	echo "$arquivo";
	convert $arquivo -brightness-contrast +20x10 "$count.jpg"
	count=$[count + 1]
done

