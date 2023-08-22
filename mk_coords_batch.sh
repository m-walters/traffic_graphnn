#!/bin/bash
sourcepath="/scratch/walterms/traffic/graphnn/veldata/"
outpath="/scratch/walterms/traffic/graphnn/nn_inputs/"
runtype="secondring"
source /home/walterms/py374/bin/activate

for t in {"0.5","1.0"} # tcutoff
do
	for v in {"0.","1.0"} # velmin
	do
		for l in {5,10,20} # tg length
		do
			runname=$runtype"_t"$t"v"$v"l"$l
			sbatch --output="/scratch/walterms/traffic/graphnn/nn_inputs/"$runname"_mkcoord.log" mk_coords_submit $runname
			#sbatch --output=$outpath$runname"_run.log" graphsnap_submit $sourcepath $runname
			#python -u gen_vels.py -t $t -v $v -l $l --runname=${runname} --runname=${runpath}
		done
	done
done


