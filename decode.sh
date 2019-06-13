
for i in 50 100 250 500 750 1000
do
	echo "Looping ... merges $i"
	echo ./$i/nmt/test_decoded_{$i}_nl_gro.txt

	# cat ./$i/nmt/test-$i-nl-gro.out | sed -r 's/(@@ )|(@@ ?$)//g' > ./$i/nmt/test_decoded_{$i}_nl_gro.txt
	cat ./$i/nmt/test_decoded_Martha_{$i}_nl_gro_.txt | sed -r 's/(@@ )|(@@ ?$)//g' > ./$i/nmt/test_Martha_{$i}_nl_gro_.txt
done

for i in 50 100 250 500 750 1000
do
	echo "Looping ... merges $i"
	# echo ./$i/nmt/test_decoded_{$i}_gro_nl.txt

	# cat ./$i/nmt/test-{$i}-gro-nl.out | sed -r 's/(@@ )|(@@ ?$)//g' > ./$i/nmt/test_decoded_{$i}_gro_nl.txt
	# cat ./$i/nmt/test_decoded_Martha_{$i}_gro_nl_.txt | sed -r 's/(@@ )|(@@ ?$)//g' > ./$i/nmt/test_Martha_{$i}_gro_nl_.txt
	cat ./$i/nmt/test_decoded_Martha_{$i}_gro_nl_.txt | sed -r 's/(@@ )|(@@ ?$)//g' > ./$i/nmt/test_Martha_{$i}_gro_nl_.txt
done