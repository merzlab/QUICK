#!/usr/bin/perl

$file=$ARGV[0];
open(MADU,$file) or die ("File not found..!");

@lines=();

while(<MADU>){
	chomp($_);
	push(@lines, $_);
}
close MADU;

#Fix method selection
@new_lines = @lines;

$param_name;
$read_struct=0;
foreach $i (1..$#lines-1){

	$c_str = $lines[$i];
	chomp $c_str;
	
	if($c_str =~/typedef\s*struct\{/){
		$read_struct=1;	
	}

	if($read_struct == 1){
		print "$c_str \n";
	}

	if($c_str =~/_params\;/){
		$read_struct=0;
	}
}

