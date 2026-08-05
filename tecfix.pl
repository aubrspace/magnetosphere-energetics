#!/usr/bin/perl
use strict;
use warnings;
use Getopt::Long;
use File::Temp qw(tempfile);
use File::Copy qw(move);
use File::Basename qw(basename dirname);

# Setup command line options
my $help = 0;
GetOptions(
    'help|h' => \$help
);

# Display help documentation if requested
if ($help) {
    print_help();
    exit 0;
}


my $start = time();
##########################################################################
# SCRIPT PARAMETERS NOTE set these manually
##########################################################################
my $dir = './temp/after'; # Path to files
my $MAX_HEADER_LINES = 50; # Maximum # of lines to search
my $chunksize = 1048576; # bytes to read at a time during the dump
                         # 65536 - 64kB
                         # 1048576 - 1MB

##########################################################################
# Obtain a file for processing and setup target new files
##########################################################################
my @files = glob("$dir/3d*.dat");

die "No files matching '3d*.dat' found in '$dir'\n" unless @files;

foreach my $filename (@files) {

    # Set aux file name from .dat file
    my $base         = basename($filename, '.dat');
    my $file_dir     = dirname($filename);
    my $aux_filename = "$file_dir/$base.aux";

    if (-e $aux_filename) {
        print "Aux data for $filename exists: moving on ...";
        next;
    }

    print "Processing '$filename' ...\n";
    # Open input file for reading
    open(my $fh, '<', $filename) or die "Cannot open '$filename': $!";

    # Create a temporary file in the same directory
    my ($tmp_fh, $tmp_filename) = tempfile(DIR => '.', UNLINK =>0);

    # Open Auxillary file for writing
    open(my $aux_fh, '>', $aux_filename) or die 
                                            "Cannot open '$aux_filename': $!";

    ##########################################################################
    # Line by line search for replacements and exile AUXDATA
    ##########################################################################
    my $line_count = 0;
    while (my $line = <$fh>) {
        $line_count++;
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

        # Stop looking for modifications
        last if $line_count >= $MAX_HEADER_LINES;
    
    }
    ##########################################################################
    # Quickly dump the rest of the file over without looking
    ##########################################################################
    my $buffer;
    while (read($fh, $buffer, $chunksize)) {
        print $tmp_fh $buffer;
    }

    close($fh);
    close($tmp_fh);
    close($aux_fh);

    ##########################################################################
    # Replace the original file with the temp file
    ##########################################################################
    if (-z $tmp_filename){
        print "Temp file is empty, the .dat must already be processed!";
    }else {
        move($tmp_filename, $filename) or die
                              "Cannot replace '$filename' with temp file: $!";
    }
    print "  -> '$filename' updated successfully.\n";
    print "  -> Auxillary data written to '$aux_filename'.\n";
}

print "\nDONE.\n";
my $elapsed = time()-$start;
print "------- $elapsed [s] -------\n";


# ===========================================================================
# Subroutines
# ===========================================================================
sub print_help {
    print << "EOF";

Usage: perl $0 [options]

Description:
  This script processes human-readable '3d*.dat' data files.
  It modifies the early part of each file by:
    1. Fixing the header (stripping '[R]' from X, Y, and Z variables).
    2. Extracting 'AUXDATA' lines into a separate '.aux' file.

  To ensure fast processing on large files, it only searches the top portion
  of the file for modifications and uses a fast block-copy for the remainder.

Options:
  -h, --help    Show this help message and exit.

NOTE - Manual Parameters:
  Currently, the following parameters must be manually adjusted in the script:
    - \$dir              : Directory to search for files (Default: './GM/IO2')
    - \$MAX_HEADER_LINES : Number of lines to search before skipping the rest
                           of the file (Default: 50)

EOF
}
