#!/usr/bin/perl
my $Help          = ($h or $help or $H or $HELP);
my $Input         = ($i or $input);

use strict;
use warnings;
use File::Temp qw(tempfile);
use File::Copy qw(move);
use File::Basename qw(basename dirname);

# Directory to search NOTE this is manual for now
my $dir = './GM/IO2';

my @files = glob("$dir/3d*.dat");

die "No files matching '3d*.dat' found in '$dir'\n" unless @files;

foreach my $filename (@files) {

    # Set aux file name from .dat file
    my $base         = basename($filename, '.dat');
    my $file_dir     = dirname($filename);
    my $aux_filename = "$file_dir/$base.aux";

    print "Processing '$filename' ...\n";
    # Open input file for reading
    open(my $fh, '<', $filename) or die "Cannot open '$filename': $!";

    # Create a temporary file in the same directory
    my ($tmp_fh, $tmp_filename) = tempfile(DIR => '.', UNLINK =>0);

    # Open Auxillary file for writing
    open(my $aux_fh, '>', $aux_filename) or die 
                                            "Cannot open '$aux_filename': $!";

    while (my $line = <$fh>) {
        chomp $line;

        if ($line =~ m/"X \[R\]"/) {
            # If line is the variable headers line:
            # s/{looking for}/{replacing with}/g
            #
            #   {looking for}: X, Y, or Z, then [R], () is group notation
            #   {replacing with}: \1 backreference group 1
            #
            $line =~ s/(X|Y|Z) \[R\]/$1/g;
            print $tmp_fh "$line\n";
        } elsif ($line =~ m/AUXDATA /) {
            # If line is auxdata, move to a separate file
            $line =~ s/AUXDATA //g;
            print $aux_fh "$line\n";
        } else {
            print $tmp_fh "$line\n";
        }
    
    }

    close($fh);
    close($tmp_fh);
    close($aux_fh);

    # Replace the original file with the temp file
    move($tmp_filename, $filename) or die
                              "Cannot replace '$filename' with temp file: $!";
    #print "  -> '$filename' updated successfully.\n";
    #print "  -> Auxillary data written to '$aux_filename'.\n";
}

print "\nDONE.\n";
