#!/usr/bin/env julia
using MriResearchTools
using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "--magnitude"
        help = "input - magnitude files"
        required = true
        nargs = '+'
    "--TEs"
        help = "input - echo times (s)"
        required = true
    "--output"
        help = "output - t2* map"
        required = true
end

args = parse_args(ARGS, s)

# parse TEs
TEs = let expr = Meta.parse(args["TEs"])
    @assert expr.head == :vect
    Float32.(expr.args)
end

# get magnitude filenames
mag_files = args["magnitude"]

# determine dimensions and array size
mag_nii = readmag(mag_files[1])
num_images = length(mag_files)
mag_shape = size(Float32.(mag_nii))
mag_combined_shape = tuple(mag_shape..., num_images)
mag_combined = Array{Float32}(undef, mag_combined_shape...)

# fill array with data
for i in 1:num_images
    local mag_nii = readmag(mag_files[i])
    mag = Float32.(mag_nii)
    mag_combined[:, :, :, i] = mag
end

# create and save t2starmap
t2starmap = NumART2star(mag_combined, TEs)
savenii(t2starmap, args["output"]; header=header(mag_nii))

