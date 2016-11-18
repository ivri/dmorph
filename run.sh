export MKL_NUM_THREADS=3
./dmorph_dynet --dynet-mem 3444 --input out.shuf.train --devel out.shuf.dev --words out.shuf.lex.vec --treport 200 --dreport 50 --layers 3 --embedding 300 --part-embedding 100 --hidden 100 --lstm --sup-train
