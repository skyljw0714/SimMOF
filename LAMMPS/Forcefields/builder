#!/bin/bash
top_file="top_trappe.inp"
segname="C2OH"

for ibox in 0 1
do 
echo "ibox ="$ibox
pack_inp="config$ibox.inp"
./packmol<$pack_inp
box_num="box_$ibox.pdb"
grep $segname $box_num > box_$ibox_$segname.pdb

file_box='build'$ibox'.tcl'
echo "psfgen << ENDMOL">$file_box
echo "topology $top_file">>$file_box
echo "segment $segname {">>$file_box
#echo " {" >>$file_box
echo " pdb box_"$ibox"_$segname.pdb">>$file_box
echo " auto angles dihedrals" >>$file_box
echo " first none">>$file_box
echo " last none">>$file_box
echo " }">>$file_box
echo "coordpdb box_"$ibox"_$segname.pdb $segname">>$file_box
echo "writepdb setup_box_$ibox.pdb">>$file_box
echo "writepsf setup_box_$ibox.psf">>$file_box
chmod a+x $file_box
./$file_box
done
