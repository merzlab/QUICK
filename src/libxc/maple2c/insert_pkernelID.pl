#!/usr/bin/perl

$file=$ARGV[0];
open(MADU,$file) or die ("File not found..!");

@filenames=();

while(<MADU>){
	chomp($_);
	push(@filenames, $_);
}
close MADU;

$a=0;
foreach $n (@filenames){
	
	open(MADU,$n) or die ("File not found..!");

	@lines=();

	while(<MADU>){
		chomp($_);
		push(@lines, $_);
	}
	close MADU;

	@new_lines = @lines;

	foreach $i (1..$#lines-1){
		$c_str = $lines[$i];
		#if ($c_str =~ /define\s+maple2c_func/){
		#	$na1_str = "#define kernel_id ${a}";
		#	splice(@new_lines, $i+1, 0, $na1_str);
		#}
                if ($c_str =~ /define\s+kernel_id/){
                        $nc_str = "#define kernel_id ${a}";
                        splice(@new_lines, $i, 1, $nc_str);
                }
	}

	open(WRITE,'>', "./tmp/$n") or die ("File not found..!");
	foreach $m (@new_lines){
		print WRITE "$m \n";
	}
	close WRITE;
	$a = $a + 1;
}

