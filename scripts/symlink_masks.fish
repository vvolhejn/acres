#!/usr/bin/fish
# When we're pre-blurring images, the masks stay the same. The simplest way to reflect this
# is to copy or symlink the masks to match the images' new names.

set input_dir $argv[1]/masks_orig
set image_dir $argv[1]/images
set output_dir $argv[1]/masks

for mask_file in $input_dir/*
    set old_name (basename $mask_file ".png")
    for image_file in {$image_dir}/*{$old_name}*
        set new_name (basename $image_file ".jpg")
        # ln -s $mask_file {$output_dir}/{$new_name}.png  # tf.read_file doesn't follow symlinks
        cp $mask_file {$output_dir}/{$new_name}.png  
    end
end
