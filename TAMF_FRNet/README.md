# <p align=center>`Robust Salient Object Detection Based on Triple Attention-guided Multi-resolution Fusion and Feature Refinement`</p><!-- omit in toc -->



## Get Start

**0. Install**

You could refer to [here](https://github.com/mczhuge/ICON/tree/main/util).

**1. Download Datasets and Checkpoints.**

* **Datasets:** 

[Baidu | 提取码:SOD1](https://pan.baidu.com/s/1LQ3v7Xc5dqkn_i-b9wXAYg)  or [Goole Drive](https://drive.google.com/file/d/1aHYvxXGMsAS0yN4zhKt8kKorVL--9bLu/view?usp=sharing)

also you could quikcly download by running:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aHYvxXGMsAS0yN4zhKt8kKorVL--9bLu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aHYvxXGMsAS0yN4zhKt8kKorVL--9bLu" -O datasets.zip && rm -rf /tmp/cookies.txt
```

* **Checkpoints:** 

[Baidu | 提取码:SOD1](https://pan.baidu.com/s/1m1GOcd1bHkIEwfOOD4y7MQ)  or [Goole Drive](https://drive.google.com/file/d/1L_wWTvscAhkhnRteg_UX1laD66ItLMMG/view?usp=drive_link)

also you could quikcly download by running:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wcL8n3lSc1pswMfDYCOBQJJjwnuhWdwK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wcL8n3lSc1pswMfDYCOBQJJjwnuhWdwK" -O checkpoint.zip && rm -rf /tmp/cookies.txt
```

**2. Train SOD.**
```
sh util/scripts/train_icon.sh
```

**3. Test SOD.**
```
sh util/scripts/test_icon.sh
```

**4. Eval SOD.**
```
sh util/scripts/run_sod_eval.sh
```

## Prediction Maps
- ICON-R saliency maps: [Baidu | 提取码:SOD1](https://pan.baidu.com/s/1sPCn3vqbalPSYOx2V9jgeQ) 
- ICON-P saliency maps: [Baidu | 提取码:SOD1](https://pan.baidu.com/s/1sKKiGqUZh1Ri8aSNMn5UCQ) 

<!--

## Qualitative Comparison
![result4](util/figure/result_4.png) 
![result2](util/figure/result_2.png) 
## Quantitative Comparison
![result1](util/figure/result_1.png)
![result5](util/figure/result_5.png) 
-->

## Acknowlegement

Thanks [Dawn-bin](https://github.com/Dawn-bin) pointing out a code mistake.
And thanks [Jing-Hui Shi](https://github.com/shijinghuihub) contributes a MLP-encoder version.

## Citation
```
@title={Robust Salient Object Detection Based on Triple Attention-guided Multi-resolution Fusion and Feature Refinement},
  author={Geng Wei, Mi Zhou, Jian Sun, Xiao Shi, Ming Yin, Xinran Zhao and Xueyao Lin}
```
