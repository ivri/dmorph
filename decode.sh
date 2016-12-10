
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/evylomova/dynet/dynet/build/dynet/
export MKL_NUM_THREADS=3

./dmorph_pos --dynet-mem 6444 --input out.log2.rnd.train --test out.nons --initialise lm_3_w300_h100_c100_lstm-pid29506.params --words out.log2.ctx.lex.vec --layers 3 --embedding 300 --part-embedding 100 --hidden 100 --nobase --lstm --decode

