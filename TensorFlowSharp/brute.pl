#
# Problems:
#   * Can not process [MarshalAs] [Out] etc, so will need manual binding, so will have to be done manually
#   
# Use like this:
# perl brute.pl Tensorflow.cs > t.cs && mv t.cs Tensorflow.cs
#
# Will produce a version that perform either invocation depending on the global variable TFCore.UseCPU
#
$next = 0;
while (<>){
    if (/DllImport/){
	$import = $_;
	$next = 1;
    } elsif ($next){
	if (/MarshalAs/){
	    print $import;
	    print;
	    $next = 0;
	} else {
	    chop;
	    if (/TF_GraphImportGraphDefWithReturnOutputs/){
		$debug = 1;
	    } else {
		$debug = 0;
	    }
	    ($prefix, $func, $args) = $_ =~ /(.*) (TF_[A-Za-z0-9]*) (\(.*)/;
	    print STDERR "Got [$prefix][$func][$args] :: FullLine: $_\n" if $debug;
	    $res = $import;
	    $res =~ s/Library/Library, EntryPoint="$func"/;
	    print $res;
	    print "$prefix _CPU_$func $args\n";
	    $res = $import;
	    $res =~ s/Library/LibraryGPU, EntryPoint="$func"/;
	    print $res;
	    print "$prefix _GPU_$func $args\n";

	    # 
	    # Remove the () around the arguments
	    $args =~ s/;$//;
	    $cargs = $args;
	    $cargs =~ s/\(//;
	    $cargs =~ s/\)//;

	    # 
	    # Split the arguments in the indivudal "type value" groups
	    # and generate a list of value1, value2, value3
	    print STDERR "cargs: $cargs\n" if $debug;
	    @individual = split (/,/, $cargs);
	    $pass = "";
	    foreach $n (@individual){
		print STDERR "ARG: $n\n" if $debug;
		if (!($pass eq "")){
		    $pass .= ", ";
		}
		#($arg) = $n =~ /\w[\w*]*\[\]* +(\w+)/;
		($ref, $out, $arg) = $n =~ / *(ref )?(out )?[\w_\+\*]* ?\[*\]* \*?(\w+)/;
		print STDERR "matched: $arg\n" if $debug;
		$pass = $pass . "$ref$out$arg";
	    }

	    # Remove the extern
	    $nprefix = $prefix;
	    $nprefix =~ s/ extern//;
	    if ($nprefix =~ /void/){
		$ret = "";
	    } else {
		$ret = "return ";
	    }
	    #print STDERR "nprefix=$nprefix and $ret\n";
	    print "$nprefix $func $args\n";
	    print "\t\t{\n";
	    print "\t\t\tif (TFCore.UseCPU)\n";
	    print "\t\t\t\t${ret}_CPU_$func ($pass);\n";
	    print "\t\t\telse\n";
	    print "\t\t\t\t${ret}_GPU_$func ($pass);\n";
	    print "\t\t}\n";
	    $next = 0;
	}
    } else {
	print;
    }
}
