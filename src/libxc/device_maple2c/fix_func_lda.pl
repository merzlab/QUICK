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

$func_name;
foreach $i (1..$#lines-1){

	$c_str = $lines[$i];
	chomp $c_str;
	if ($c_str =~ /(\w+)_func\(const\s+xc_func_type\s+\*p/){
		$func_name = $1;
		$b1_str = $lines[$i-1];
		$nb2_str = "#else";
		$nb3_str = "}";
		$nb4_str = "    ${func_name}_func1(p, r)";
		$nb5_str = "  else";
		$nb6_str = "    ${func_name}_func0(p, r)";
		$nb7_str = "  if(r->nspin == XC_UNPOLARIZED)";
		$nb8_str = "{";
		$nb9_str = $c_str;
		$nb9_str =~ s/xc_func_type/void/g;
		$nb10_str = "__device__ void";
		$nb11_str = "#ifdef CUDA";
		$a1_str = $lines[$i+1];
		$a2_str = $lines[$i+2];
		$a3_str = $lines[$i+3];
		$a4_str = $lines[$i+4];
		$a5_str = $lines[$i+5];
		$a6_str = $lines[$i+6];
		$a7_str = "#endif";		

		#$nb3_str = $c_str;
		#$nb3_str =~ s/xc_func_type/void/g;

		#$nb4_str = "__device__ ${b1_str}";

		#$nb5_str = "#ifdef CUDA";

		#$na1_str = "#endif";

		@tmp_arr = ($nb11_str, $nb10_str, $nb9_str, $nb8_str, $nb7_str, $nb6_str, $nb5_str, $nb4_str, $nb3_str, $nb2_str, $b1_str, $c_str, $a1_str, $a2_str, $a3_str, $a4_str, $a5_str, $a6_str, $a7_str);
		
		splice(@new_lines, $i-1, 8, @tmp_arr);
		#foreach $n (@tmp_arr){
		#	print "$n \n";
		#}

	}

}

#Fix static function selection
@lines1 = @new_lines;
$l_offset = 0;
foreach $i (1..$#lines1-1){

        $c_str = $lines1[$i];
        chomp $c_str;
        if ($c_str =~ /func\d+\(const\s+xc_func_type\s+\*p/){
                $b1_str = $lines1[$i-1];
                $nb2_str = "#else";

                $tmp_nb3_str = $c_str;
                $tmp_nb3_str =~ s/xc_func_type/void/g;
		$nb3_str = "${func_name}_${tmp_nb3_str}";
                $nb4_str = "__device__ ${b1_str}";

                $nb5_str = "#ifdef CUDA";

                $na1_str = "#endif";

                @tmp_arr = ($nb5_str, $nb4_str, $nb3_str, $nb2_str, $b1_str, $c_str, $na1_str);

                splice(@new_lines, $i-1+$l_offset, 2, @tmp_arr);
		$l_offset = $l_offset+5;
                #foreach $n (@tmp_arr){
                #       print "$n \n";
                #}

        }

}

#Fix paramter selection
@lines2 = @new_lines;
$l_offset = 0;
foreach $i (1..$#lines2-1){

        $c_str = $lines2[$i];
        chomp $c_str;
        if ($c_str =~ /assert\(p->params/){
                $nb1_str = "#ifndef CUDA";
                
		$a1_str = $lines2[$i+1];
		$na2_str = "#else";
		$na3_str = $a1_str;

		$na3_str =~ s/p->params/p/g;
		$na4_str = "#endif";

                @tmp_arr = ($nb1_str, $c_str, $a1_str, $na2_str, $na3_str, $na4_str);

                splice(@new_lines, $i+$l_offset, 2, @tmp_arr);
		$l_offset = $l_offset+4;
                #foreach $n (@tmp_arr){
                #       print "$n \n";
                #}

        }

}

#Fix define statements
@lines3 = @new_lines;

$define = 0;
foreach $i (1..$#lines3-1){

        $c_str = $lines3[$i];
        chomp $c_str;

        if ($c_str =~ /\#define\s+/){
		$nb1_str = "#ifndef CUDA";		
		if($define == 0){
			splice(@new_lines, $i,0, $nb1_str);
			$define =1;
		}
        }

}

push(@new_lines, "#endif");

#while(<MADU>){
#	chomp($_);
#	$str= $_;
#	print "$str\n";
#}
#
#if ($str =~s/([A-Z])/,$1/g){
#	print "$1\n";
#}

# @arr=split(/(\s*[C]\s+(\-*\d+\.\d+)\s+(\-*\d+\.\d+)\s+(\-*\d+\.\d+))+/g, $str);


#@arr=split(/,/, $str);

foreach $n (@new_lines){
	print "$n\n";

}
