pen = [0.4, 0.8, 1.2, 1.6, 2.0]
lm = [0.2, 0.4, 0.6]

for i in lm:
    for j in pen:
        fname = "conf/decode/lm_noctc/lm_noctc_lm" + str(i) + "_pen" + str(j) +".yaml"
        with open(fname, "w") as f:
            conf = "batch_size: 1\nbeam_size: 10\npenalty: "+str(j)+"\nmaxlenratio: 0.0\nminlenratio: 0.0\nctc_weight: 0.0\nlm_weight: "+str(i)
            f.write(conf)

            cmd = ". ./run_mseame_tedlium_aishell_lid_8k.sh --stage 12 --stop_stage 13 --nj 200 --inference_nj 200 --ngpu 2 \
                --asr_config conf/cond2/cond2_nosc.yaml --asr_stats_dir exp/asr_stats_nbpe8336_ms+t+a_lid_hop160 \
                --inference_asr_model 10epoch.pth --inference_config " + fname + " --use_lm true \
                --lm_config conf/tuning/transformer_lm.yaml --inference_lm valid.loss.ave_10best.pth --test_sets devman"
            print(cmd)