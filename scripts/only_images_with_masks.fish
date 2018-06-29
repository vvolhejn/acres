#!/usr/bin/fish
# Copy only those images which have a mask.

for f in $argv[1]/masks/*
    set name (basename $f ".png")
    cp {$argv[1]}/all_images/{$name}.jpg {$argv[1]}/images
end
