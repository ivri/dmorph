
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/evylomova/dynet/dynet/build/dynet/
export MKL_NUM_THREADS=3
./dmorph_pos --dynet-mem 3444 --input out.log2.rnd.train --devel out.log2.rnd.dev --words out.log2.ctx.lex.vec --treport 200 --dreport 50 --layers 3 --embedding 300 --part-embedding 100 --hidden 100 --lstm --sup-train
