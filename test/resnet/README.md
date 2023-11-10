<h1> Instructions for running ResNet model </h1>


- <h2> Prepare SSD disk </h2>

  <h3>1. Setup env variables </h3>

        TPU_NAME=<Your TPU name>
        ACCELERATOR_TYPE=v4-8
        ZONE=<TPU zone>
        PROJECT=<PROJECT ID>

  <h3>2. Create TPU </h3>

        gcloud alpha compute tpus tpu-vm create $TPU_NAME \
        --zone $ZONE \
        --accelerator-type $ACCELERATOR_TYPE \ 
        --version tpu-ubuntu2204-base \
        --project $PROJECT

- <h2> Attach SSD Disk </h2>
   TPU only supports attaching disks in read-only mode. Therefore we will follow
   the following strategy.

     - create a fresh disk 
     - attach the disk in read-write mode to TPU VM node
     - download data to the disk 
     - detach disk
     - reattach the disk in read-only mode to all the nodes

      [Detailed Instructions to attach a persistent disk to TPUVM](https://cloud.google.com/tpu/docs/setup-persistent-disk)
  <h3>1. Create a fresh disk </h3>

        `gcloud compute disks create lmdb-imagenet \
        --size 200G \
        --zone $ZONE \
        --type pd-ssd \
        --project $PROJECT`
  <h3>2. attach the disk in read-write mode to TPU VM node </h3>

        `gcloud  alpha compute tpus tpu-vm attach-disk $TPU_NAME \
        --zone=$ZONE \
        --disk=lmdb-imagenet2 \
        --mode=read-write \
        --project$PROJECT`

      Login to TPU VM using

      `gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --project $PROJECT`

      -  Format the disk
         
         `sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb`

      -  Mount the disk to a path

        `sudo mkdir -p /mnt/disks/persist`
      
        `sudo mount -o discard,defaults /dev/sdb /mnt/disks/persist`


  <h3>3. download data to the disk </h3>

      `gsutil -m cp -r gs://imagenet-lmdb/imagenet /mnt/disks/persist/`

  <h3>4. detach disk </h3>

      `gcloud alpha compute tpus tpu-vm detach-disk $TPU_NAME  --zone=$ZONE --project=$PROJECT  --disk=lmdb-imagenet`


  <h3>5. Once data disk is created you can delete the TPU VM </h3>

- <h2> Create TPU VM for training now </h2>

- <h2> Setup env and Run the training workload </h2>

      <h3>1. Setup env variables </h3>

        TPU_NAME=<Your TPU name>
        ACCELERATOR_TYPE=<ACCELERATOR TYPE>
        ZONE=<TPU zone>
        PROJECT=<PROJECT ID>

      <h3>2. Create TPU </h3>

        gcloud alpha compute tpus tpu-vm create $TPU_NAME \
        --zone $ZONE \
        --accelerator-type $ACCELERATOR_TYPE \ 
        --version tpu-ubuntu2204-base \
        --project $PROJECT

      <h3>5. reattach the disk in read-only mode to all the TPU nodes </h3>

        `gcloud  alpha compute tpus tpu-vm attach-disk $TPU_NAME \
          --zone=$ZONE \
          --disk=lmdb-imagenet \
          --mode=read-only \
          --project=$PROJECT`

        ` gcloud  alpha compute tpus tpu-vm ssh $TPU_NAME \
          --zone=u$ZONE \
          --worker=all \
          --project=$PROJECT \
          --command "sudo mkdir -p /mnt/disks/persist && \
          sudo mount -o ro,noload /dev/sdb /mnt/disks/persist" `

    <h3> Install torch and torch_xla </h3>

      `gcloud  alpha compute tpus tpu-vm ssh $TPU_NAME \
        --zone=u$ZONE \
        --worker=all \
        --project=$PROJECT \
        --command "
        cd /usr/share/
        sudo git clone --recursive https://github.com/pytorch/pytorch
        cd pytorch/
        sudo git clone --recursive https://github.com/pytorch/xla.git
        sudo pip3 install numpy
        sudo pip3 install torchvision
        sudo pip3 install torch --force
        pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html --force
        pip install lmdb"`

      `gcloud  alpha compute tpus tpu-vm ssh $TPU_NAME \
        --zone=u$ZONE \
        --worker=all \
        --project=$PROJECT \
        --command "date && LIBTPU_INIT_ARGS="--xla_jf_auto_cross_replica_sharding --xla_jf_bounds_check=false --xla_tpu_prefer_binomial_single_phase_ring_emitter=true --xla_jf_single_phase_ring_max_kib=40 --xla_tpu_megacore_fusion_scaling_factor=2.3 --xla_tpu_nd_short_transfer_max_chunks=1536 --xla_tpu_megacore_fusion_latency_bound_ar_fusion_size=6291456" \
        PJRT_DEVICE=TPU XLA_DISABLE_FUNCTIONALIZATION=1    python3 \
        test_resnet.py --model=resnet50 --datadir=/mnt/disks/persist/ \
        imagenet/  --num_epochs=46 --log_steps=312 --profile \
        --host_to_device_transfer_threads=1   --loader_prefetch_size=64 \
        --device_prefetch_size=32 --prefetch_factor=32 --num_workers=16 \
        --persistent_workers --drop_last --base_lr=8.0  --warmup_epochs=5 \
        --weight_decay=1e-4 --eeta=1e-3 --epsilon=0.0 --momentum=0.9 --amp \
        --eval_batch_size=256 --train_batch_size=256" `


<h2> Comparision Numbers </h2>

[Please click on below link](https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1&title=resnetnumbers.drawio#R7LzZtqNI0i74NHXZazEPlwIJIcQ8wx2jmGcQ8PTtrr0zMiKy8q86fXFOdXfFihWxxUbgbm7D95mb%2BT9wvt3vUzQUSp9mzT8wJN3%2FgV%2F%2FgWEogVHgP3jl%2BLrCYMjXhddUpt83%2FXnBKs%2Fs%2B%2BIft61lms2%2F3Lj0fbOUw68Xk77rsmT55Vo0Tf3719vyvvn1rUP0yv5ywUqi5q9XvTJdiu9ZkMif18WsfBV%2FvBlFvn%2FTRn%2Fc%2FH1hLqK0f%2F90Cb%2F9A%2Benvl%2B%2Bfmp3Pmug8P6Qy9f3hL%2F57Y%2BBTVm3%2FDtfGJs46cSbTQa7nuYhiRXy%2BX99P2WLmvV7wt%2BDXY4%2FJADGPcAfy%2FYjKm7LpqUEApKjOGv0fi6Xsu%2FA7%2BN%2BWfoW3NDAX3BRUr%2Bmfu1Svm%2F6Cfw%2BzfJobZafnnBpyhf85tIP4Go0D18LmJd7BsbMfV54%2BeMq8scV%2BKhoif6BX74%2BYsLQvf6B8aXLaeYbed5f%2FQX8US2nuDkv8FMD%2FxFO%2FhKA%2F68l6WcI%2BGEpnOZmuCbhr3h6JTk3zHExCZ4X7mHUzqVjwyS4ID1RaKYmqE2AWYYY6tTRXip130ss7gTRSF61aSCoaat3ZRHkwyYe5T0nJQS59e%2BrGR1wOTDuJB4xz48UVdIvGXwGV88WXq90NtvxE8dX3cv1md3PLGZbfdZYEu8wXPPDmQE3k10F%2FsXjkrleLvzF1n2Lf8M5XlhBgLO7hD0DfnG5Pu4Xx7tcuMtelvCOi1JHHGdcLvX19pTB59djN4PL7XK5W8rlNYPHcdX%2F1mfIgsJT%2B3lWdefEDycYcK1GwCS5yKI%2B0hJE6cS7rmuTeV2ZkyH2r8uIgD6RDcoheNXNkU8oe1ih7q9Ya6dbt3UpjZOd39FkmCNN6zVrPEWb46Z%2BiNPYTIJXhNpH%2BBzj4CnHDeTXg7vmZBgEb4l525ZSYwgEx9syy7SHNOP55i8YSpU%2BiTvcPm8o8G3nBm5c6XOowbPws%2B%2BDXJFxuE7WOecHeSL7%2BbXwSHYmq6viavT1YvB3HJb6JE9yfu8C8BZ13LHssqkRmDzHDtPMGy73avKrnoNHUCfUoGTxYhb8esvocYi%2FHuPdJGnuaFbPmiRRbkK4bRidT7LSiSn4g2Yj4j7RuJfX8q77G9ZqvtzgOdnGQ5KtbTOM98Lw2dD1dfCGlDI6M5gTQXQ7UWJJsmjoh5bG%2B5eIOPaZrBh11uTXvIQsSkqKdH2%2FbBj9nCyX6LeHbIKVfRNZCCUcPpIKioS5Xs%2F9hXc0HefrpkvJ6TndTOHCg9mGPc%2F8h2NFzdfs0SmWau%2F2SBpjpOCXMcxLFyW9ZvhI0T23eJM7NDuaYnrPMExfpc%2FYQ5e0PespTxLgf75k0%2Bj5Ps6vW4xSWdVTovlqlPBr4ODvkq8sTqKbmCTQtsJlXbeNJdfpfICPtDqiUObnEuPQQg%2FMp1OPIggjoukHQdMVv0ElpKgOrIRQ4%2BiybamzNqeGGfeX4%2FFeAleOJtlt9buQNx71mh1om0h6jGq8SeYum7dpWdPJQ5ZZU219fpNDBZtLB3xvxXVdT9MVITC1iicoyw3fcujXRf8g8svqGcaCs3gRxSp%2BrusKrARFs4RBULzziNe5kpXFPctyD%2BVXolbDigMZxY8TU7NcV1XwQPR4ZtEF0UPylfkRadpn46ZZ5DN7PE3DUMv0UGMxrd5N4Woeoo9zr5ePaRPiaI1IpcAANFOGMvKgVgZsMg1EvK1ZkgFvzJn2VHuysHToKuhPunyNlHk3wO0b9cTicIGOD%2BlChi5l4SBmHcQY4dZ24s7kvvse1AjYdDk89iZWaWZDpZY9Y4bMvhbv6ct3q8TjDT%2FTeUFFsUk9KKPZZeFylChUufe4Z1s8zayTx%2BS9wMhkdMOVTtYcGk8lf%2Bs0tEko1wjD8UnqyNaYSZ9QMTcGgZQbNC09u7ijj3W6qZnD0K8joNshTDNqMcXmut33HBgy1BIMJzXGre4fY63DF%2FAPGw1%2BtpnRrjo8UvA0h29FnZVD2q%2FX53oH4q7ATG6EDOCzGNPrw5vSFbxzHMdtBrd57aHQNEWNm1q8kFjbHNNKsqbTsKJV%2Ba3x7fXmPpHjDCaWqu6Vp5DkHegzHBL2Xg4r5fAjJ59bDVSS6x7smsj3ZYiCDlHl%2BFWvvnXtTU8xfG%2BC9gZFIqybqHQP3ZJ5P6DkEnxxrVBKy8HkOAs4wHjTxTOhvJz04SQ5Fs40Lt13y5gIna7l6m8TzZB3yc%2BB8sUjhHLCY%2FHoOCBtOYRPpJSKgJo9gd%2FMWXh5JxpQ%2BVTrLBna8NUUqRWJVRPJ%2FAPes5AU9BBy%2FmKOl0WG82jKLWI9X1mpYonviRgmZydGogdJcRp64OCPkEZpOO1y7INvHgl9lmW5UqwkLjZPDtqtu6%2Fca5CfNng%2BDyfH9NLKID0MN1glYPCSrV7jKSdmT9Zo1Gu%2F12l5o%2BMeIpjLyj3z5XLasvDJYX%2FGDiuoqj2hAFXwao2RPm%2F1PUtlbPY4GvJNeHZYoMdGwkHho2E0nmsNyhI%2B988apXn6PhzxjnC94Kqvw2JHB1Xt4QxxGGzkaPG25RbVyTsZzZk%2FbOCzOKWHujaZwBCb27UFKsIataXtydyYjXmya%2B5LGnuUGk0eK1xifPdv9sYdqTnX5sgffo9tZpeLe5nNE0VmcYci%2B0NPMI2b8XgCiyY%2BvRIouK2EiVTtodjh%2FuAuOK4wNFYQDXQG8LnQ802fyEVCxEKyup57Tm4SKbByTog3bKVJSZJpIDA8XnCBCZ2zkR%2B57x2%2BTm1TzRtbTUWMRaEfW9I8MpYwoyBju%2BOhcqLAF66%2BGaw36a4mlbpqZCg9gf%2B469CvBCvUE19BMh3GNZJo2Mx67e%2BOTqDpY2JcHJq7ZRYh9MFxTuqqvsI3myxPxaK6pQ9B3AcBgCTpXqnqh5CR6HroqrYcIZa9QrPqXvVFSgBMwO2%2BmV%2BobDS4erl0pr%2BbZqC0rWMQVDTITPJy4gHFRb1HUtEpKknpUHIkq95O8SOrHiTlzDoZAO3m6iKTH7tW5jAAmTAKOMcjLmRLJtJQ7iqCArhFkHc8nB0%2Fny1VRGQpwffXQhy5TCZVnGRzz24ahDcnnZ4HTyW7c2DJcmjWBg1ODXIo0Ph%2BxxinmVeVDSi6EbjRO2oUHz1Mx1PPKElUs5%2FaXB9PucylO5dU07GwNU44Bp6LOlMHK058JLu%2Fg7B5T4dfEYI%2Bkh%2FDkIsdn3Fv6CtfXSU7EvtpGNNkTKHFm2kyBwced2XhsFVb136QOBDxSezhB6YcDvWdzM5eI9PSog%2BfJ1MH4wsJNZVX5G2Nk0syKSiyKN%2BeSH5Jss05AF7WAgkjpRSo93KXX%2BKzjgCYA%2B6httcr80i%2BooZyxw7EWE1mO9BVOhp9parlrKQcT0d%2FJCde1%2FsvO3aS9PnxTcJ0J1DFFmUpSNsjdFGMWYl%2Bbcoa90W53DCGoEk9pUIeZ98jy%2BoPZn3F1%2B6cUgw37yjb9J0vQ8PMCVNKJ0ApaVSPu72mnjcJCBI9FO1UbPUMIPZNAogZzJKGgKu3NnFeNGPtpLwhdx3PU58h4eKu3mGjikzTRA9ceB6xdtvdhsVGHITUNteb1tqS5Li4GyjxziLJo%2FOBqSmSW0wtgA4k7KaW7VLS0wBKpe%2BWDp03srBpOCQxQ4jGc4wvZbWW9VNbfWjSo0sAjeS0RqBan8yiqW6L0EZqZnaqalPlgkXg6hNkKW1vgrTpdaZ%2BYC10wyBESMcu8x3qIfh%2Bft%2BPVPKpu37rvI8siY7OHiuF4gDRX2cqnYIXfdmfMB610pJETLich8yksfxxLyrQJkHbrat83%2BNKLmOSnjXLaoV3BBWdIOa9%2BHp5zqzL8mYh4cpQZHszrJ64sa%2FWV82YC5piJBiTJyRhHtaprN4kNl79pQbu2JI7jiUmbq6CKE4psyBl2bl0XZnSPSXI8LnI1SDgF5x8L9jW5GaV9hfkI71Vz5QSysASn6zPECAgZpT1fNqDS%2BlgQcc1i7X7zdj4ZDBrea8sh1KPI%2BEtfCVjLNlY3EmxDHz54DtqddHQf%2FowhtomXBAV3%2Beks1ymMGnZ2lgzvAvCCCS50GuR5u%2FnKVzuh66PWg%2F0NCVGtIlnpoQA0jtYPDgMjNBrGP1KkkVs75GytJJLPk%2BZOstJprCqOd3JuK8BUuRt0t2Yu%2FJj4E81aCH23b4QzAIxGX1POjC%2F5jBnoZOuVb%2F5dUR6UcbulDX7%2FkIWFu3IqxEaZ6TL7O5j%2FHoUmJa%2B3UGUmsurXPrXPKcHe2Of8XGGMy7QRJtVVgdRR4OO62pi0fYy%2Bmav6obd2HruTcfunnmK9NC3GL3J6gdNg6C%2FrMNmbxVBavWEX7crv3UQzKHyZK8rHqE5DvWqjLdDXZlksqc7lgxpp6Qktcbv0x5oH2UTv5nb5irU3mYk02G%2B%2FAfjid2lNRdalwMEefJbbUv6g0r6Tjg8mTlQfO3WVTfFC%2B4j0Ou8S0%2FuiPy92N0KbLeE1huB0CCj5H2WcPas0EYZo9CmvKqlpdmDMESl%2B4OQTLrAHCRZa2OhXw9lrQq7N6za8YHRpjb%2BGL0ylFcz5e1XS09Eo01iD62sawcyHBgXOh2kxqSnvi1Pudge18Q7CPyWB2XazaO8jdttYHGv6qtZLm06xUN8ZxygUueJYXcN2%2FYTPYC8NawPn5DtDYQPoBeqfrQsXpORIliaIz1mK3Scdah96Z5%2BeIpZNjVJPMQdQB582TkDoeIgYEI%2FYgu8peThDjiQzyYl9sTyW8gnNUa%2FJra14oolwHd9oqG3RA%2BTU6mZLyuOTVbOOxj6K4Vglkoi1qRIpXlSgGeq47s5%2B0E%2BlN12iioP1M%2BrKaaxZoRNwue87VWurUZau1JyZM4H%2Bixswc8GGZLAmrgYs2ZcOtzFStnUyIn3%2FeaoN8HdzBahqwen6T2bVd0QhtO8aqbyrijVcg8SYCZnOVmRJmW5JzOTFnq6hg7PhHSm2rMOLrj%2Bqilrq1XxgAkBrjRFT0cqRPO9Lk1W5ND2jELrNIOk7H28NMWPjgbLdr%2BayFg%2BU4ihquhpEqVAYDbweHr3fBE6pVjCUSOq2i30AwYO%2FOlDzJ%2FuHZk0InmzgCR1QLm4F%2FZMKwSQ05plCUmzyMimKY8gF5oozTTfz1xbTpWlCSIZryU961aduXuMB5e5iU0uDJIhqaHwlUkzu6BUZvHB6fn7%2FYYrM5G3tIh0v4fewfcKUsxfXXOe1eMWI4mWNEBsnYP7LkTUwF5v4SOhwqjMmVY2%2Fe39NvKFmV5nmCKhSDJ1buJhkizq7D6ddbpbLSaiUmpa1EF7cqssHZ5XeREcVjKK0jrGeb%2Bwm1V57kofshLQ8VQ7DnALFXmo9W0xmPYeJIZrr56alOuhDEFpiLE2GxjdNkvm0G6VA5rFSQiMKacM3HQGUb%2BTa6HShcQGbjIVOH06m7eCLOUSk0iSeBeBNzc0Ka7mzl6f6w0bEmKz6%2Bjd5TindFQ%2BVRBzxcXZytSt9M%2BFfAvNtomk%2BoTYgl%2B6CZARRNB4AX1dnCdFiJZzIbmbIPT80whuQi8MRnC593fODC7PXuDM6FLWd46LuMi5X7jLhaury%2BOTjgtgOg78uUc6%2BHS5HNzrk44zlKCf4S%2Bu0v1lOJeLdNl5mI578Irj%2FT99Br%2BP5QoYEJlGTKSLGKNp2uWi4GJp6jqCIIyPTeEKYCDFZhNYSZLNYyRUsEih0mzryPodBFK44STbW%2FgTPdjlDoY08qTrYrwqVng0CQeJrGe0JEuS66JATlVVAXEduTQgVChkEzvt%2FTsMV5EvoohcgQPvKjy2JcjGfD%2Fbcx23ltft8tIGKzVcz70v07jXRHK%2FFtBQDjJhMCbLlVJPiSYEsKYdppkgojsHKAV%2BrVxixyBJb7sYZjHTqBsv7SY3Jd4CD6nL9pErrffcvDcRKi9DwTqAu1QSvj%2FkMYrjXshOZzTQT3HD40p9vR8c75HW5rM9Bp5o%2BNFIvYXxHTyEcPEGl8jbx83SFWrxpji6n%2FGCUAruj%2BhNiruTZPmeobVqUGiWYnn4muozx7Y5JhYwf3OWHoFhIL0GMYLHeuMUX7DJHOqzo2lZR4n%2ByWafzN07UIBICR28Q1lkD8WS9CjqTgPeZ2edLebAqo%2F14gpNcUQ5D1aUOgvnvSdFCMmKfi2KLAGc1GvnWDT1IdrEc8Hi5205lmSLTaAtz9fquunrSBLElggGS4GTn2v6qvlWz5l%2BxEazEmV53o0FmCyGzkeuwwiGLLIVrFMPZq63p1L427z5MoaANeB7uXH9gENJUhIEyPWWZzmuVXXSEQvpUY7vYaHh%2BSrL5oLJV67iekC07QWD8%2BaIT455YUZpmvVfn%2Flw3hEbzJsBJv689Pz4Pk6pPs2D4YHe9%2BXr%2F4SV%2Fb%2F0GZ3ciTBwb0qUzHHl5X7K8eymP9LyGl%2FenHN%2F9TOpEIOLqkntJxX2Vi4u1z9lzdLw%2BWZdXrxbAImvkYpPvXExnl7xvskrJSHTw%2BCMyCrf9XWhSKR%2FXAQj4suzrsDH2%2BtxaYKA5%2Bm2SinqdrnxzRj3PEzy4dHixXD5dbF9XRRNAT7arswPj0WRPoKWvi2Ucrf1O6tt%2FokcgaLcZ5%2F%2BYgOTHPSfn5HYKe40%2BpYuAfd0cEdA18rsGfzZWJs6NRh9Ko7vY8eHiFOEsazwmcq7d2TbV8%2BZrNmMjkIdIFvPQ03gDAeM5BV8qj%2FzHtb%2BOT0rD22CM9Gul3enkC%2FWdtC7kaungbSiWO2D7wqrPzV0oB5ZmT6P28W5t%2BM6X6O59WQtj4PgvencG4lCdYrh3PAK5lra9%2B3KDfbffJuL5jvPebE7uplA%2Fu5xDmyUmc2VMSKsWYVyb8B2VMGNgzuQrdImeR6HXRioP%2FsUAfiU5%2BDbX%2B8LVYwe0JFticRShtuEnpJjbg%2FCMOrys03GoE%2FXBICj7d6Lqmoz4SruDVUrr5lj%2FkTuwJBjoVmcGIKNKF2NSv1e9SgegRaQ7SOW6iOP7%2ByBYssQKfh3olXVLaSVR2qdFGjXfzOXb99yb10Qt7pMmv%2FqQ8aPD2mpMVnK0yO85ap0U2e9XhdXGscnzfHt1Z2ahrrlTrX5l5p73Ybry%2FXumq%2B31M0PIDO1tcYEHvZqOJ7ySjdfabjvj0md%2F%2FmtwAfOc%2FGNF%2F9CGu718HPkHBwD%2BFZEEKqHGAMJGt8fO0n8QysN4g4o4o12kcelVyUj4AX8xywBljg4jE0O5%2FtjeEAQzCa8cb9YtcYV6WlqaMpfnlcLoIwi%2FU8dbyDmOknnAUKGuDo0uAIBl8DWWigul%2BzzPYQNzAu%2Fodo6TK%2BSRiIQBEzxM85QER751SYQ5%2FEe9Yux0n0ZPCBg1rr98bIL4Os%2BczdvhlExrPgyX9uln6ULnY2PFyeS1HYpuD%2Fe44D36CeaFfyF%2Bfijgej5xz0XS5a%2FccRnvvZu3d436IOE4%2Fao3tDDKR1fy6%2BByeTb43tk5v%2FnRya4YaDhwZxoinhtLPkM4wUj0M2fUKqwum2is4qrz32zJ5RhlfZ65Sbha89c354uzFfJqMWu91D%2F%2BU38OM0cRQ5Nd15FUVS1IRMlb2r2Ae5lDUUaK1isKJfZlytbPRO0ZdczDLcJoeJRJrPVX164RD1a%2FqZXA2M9boTvLUYnr%2FUNRAxn9eUeWWYt1jpva89w7QR0OdKxHcYEJo3itugVTb9XrsLiq%2FiHh5VexZWg8lvJaQI6OdJBD67vTyLMoBZu4FChopifaHQtXu%2B9PkJVGWY8PijkavbIIVs4xJArnm%2FqS7lf%2BRIx844aPBT4nrF8XKA9ZNP7YfhFOpKX8QZwV2W%2B9s8eWNp72rp5k63aA0Isi%2B4D%2FJy%2F16dWQQJnr3i8vXVl%2BuChernZnCW%2BYb5EFXhFnrzedcr4xT%2BxacMntGW0K1dkORgpcsKNV3a6J0H4OO1OgDE%2FfWKR%2F4D7AHOk3pHW%2FEhTxtRaoU0C8R54HT80UmRjNGbAk7XGGt8kf3qMJl55hp0ACFcCTKvyeG%2BOPTjT9QyMZnUmmLqO1NKqpPfAE0%2Bgvjt%2FIwEQXst2PcSOJMNcfT9uVzaTFUBfMUq9V3Tkcs%2BI9iahqWzIpFkWYOnYw5bWZiFEaZshsgU09d2WnI%2Bwjlimodgcq4K6P5KnsfLdTsxGoWYov7btwkKIiu07igpCqsNtvOfgQeS5rTHcLBYouJs4gRGYYGWbd5rntjl0P5KhbXyQStOSDJUWMYyVUeAAQWEZq%2BPbrevQrTHVvKHIDLsm3W%2FWlUYorxSmT2q2OQntyfLybOC%2FakDlOsZNUNtcE2UEbgyRfPa8lByYxmYA00MukuzvlMLVhfyr1U9IMnUZQAnsJsfO6smqoBpqh5i3G9yT0E3iFsjvbLIvtxFbJi%2FukadAKe4F092bcQFG6PfGa%2BOIbRai2buLuWVk9WD9JEXw%2Brcq437c6pELM7W53K38YKq%2FjkTGSo4LrNfPK21%2BxYhftPMhey23YHSsduI0jllsGG%2BY6Gvewff%2BRMKGYUjWON4V1%2FTtP4fyt9Fcpy94p%2BnqZ%2F87Z%2BoqP9mNilqp04chzKKvyhTPT4RjDH7yiYa%2FKJdIaa1H5JErPdkzm6ZABZMsu%2FNd6FysX3ze7SZP6rKuDMrQ%2BJYrDMOy5Mk%2BWF6tAqxdptjrFodHlsdf5ngdKqx88CTvdICIvELweuRhiIWjJXWqCw%2F7YhOM%2FzBK%2FfJayV%2F0Fs5RkvRYyvSQ5X0urpFSo43b2ykYVa9Mo%2BJ5dBos8dbk%2BGwc0vN3bZkolS%2BGyUNdi7mvtfvT%2B15b9LijsVp%2Fdk67hanvlSHEv2nMqzDh9kZHTeuFG6jgJ%2B%2FqisNDzH%2B7%2FynzhvabVhccf%2Bys516PbsuY%2Ff3OdLVMMu0psTDj1l9u3fCrrXAW9p6mrr1X6knW50n084ZNrICpVyO4x2rbOp7sDrVQsln8IL6ss6tYyeGkv6zbQ9H15YWoZ4p7ZawXbyQOlynC6u9SmA2hF0NsEO6n%2BPhaokBC3%2B9X00KCP4AoccTAk5Mb0NgsLg%2BKOnjC53YUqfX%2B2V%2BNnyI1z%2FQX3xupL%2BWMZoIg3u8zWU1o0VS3fr02%2FfoPg7lmAVk9aFHsD1ezjU%2FiarE3kwMPRTLuzjOPh%2BfdLo%2F%2FGSXAd3vQp%2FWvuz7XqZdefosJiDOIcff0K6wJ5taS4pf8q2WpIHyHYR97CGoQiYYGKzev7U9xUEQzoDnB7Dw%2F20b8g%2F%2FrCLaVfT5m32VA2E3d7BWLf9WJ94u7oiaYElQfl7o9ru8Pi%2FzZTvzl8mGAudZlIPozDFLsKNbKCJJpZVGQg22fGMrcutUA6tmXl1vejQx%2FEwAvWefR1boW1rIMdarhUQDrqDBFHstyXRU%2BS1NJ%2FhsPlmY8x5E9juPHiZwSqbeV0d9o%2BcXbf7GQ8AsFuGmahh7rygVOo2TsqyhYeBTTQt77fUbWQajoEnttqWmaIiM0%2B8l%2F9Au6F02Hae7F%2Bx0b%2FDN5yceLS2OCyVQ1wbXTf%2B1ISf6EDwXauKwbzx8MgqIkPaUtFofbQGmVcTqX8XdLkXW0S8aHZGX0fq%2BaYxDImVq36Qxfe4Acn0CJK8Wltk8dYD1mGcu%2FwXJX3q0m%2Bn3Vdf0OZphFAcZL64%2BMXx4dZHaGM4MjVKrLUb8dMG9VtL%2FLdm2CQvPzxyPIggEguB3IOnlFwxh1kt6SfXS9VnuPTOPxnYckZfNn70ihoxDeKxJoIWbDapAeCH7fHdbzu64R8mXD79uDIw9soQFmoRecPoMr93r3%2F%2BMTb%2BVz3J3JXStBt57rW5sAOgOjLuZ0R3D4FBpMqjtZZAqKu%2F3bmq0AOd2LF5FGzRBDU%2FVLsRD84ILE6nwgS%2ByHO%2F7wnnkPAvr%2BJ19GwAvxtgd3kY%2Fyg%2Bb9Pn6EWAWtdgswIz1kGqd2ZHIex3I%2FQ6XTPc%2FLQ530bWj%2Fj%2BckjdF90QPs%2FspjtgUQ7Z1qdDQvZZFvtLlYt%2FDniLRFu2ddfAiKXHfnS3yG%2BxyxwjKf8icVDFaOonWaaaW7tuevWZbf8Ak2g2n7JT%2B%2B%2F8RN%2FnYlBZRllsYhY1NAYe73b3D347gJeUKq7ezFOvJY9%2Btv2jv2Xj8BdK%2B0keIHMIdq%2FRonH5LrrlyRjOXPyESgDAupMmR%2Fuw8LYZxC4fRqT4AbYx%2BDo3BFeP3d3huTF4TXw3J%2BvVtyrH9y9%2Fiw%2FhIPXx8e%2FjPu0gHNIh%2FUT9EbUmDAlX8e0YP55Mb%2BFa56Vp9s2888yBU%2BbP5fR%2Br%2Fjuy%2FI%2Ft3RybH03ucPz6pgQk37%2BmgbB4zkXL3dHBh6aT6CAI12nxYkEceXQ4d2eguxin9Eqkh6ve4xZcbq1IXrLJYTOr3gPjeHgL4XJ3U7jzjLy%2Fr%2B9b%2BT7xVsW0HYKMzlUK8ADAilfgeLmEjIKWHTa7R%2FfUmmEh1Tjwj%2Flm2IEN9Lo2cp6sr%2BwPEuPBSGMxRW6GukDMW8xgGIgNy0Nk3oY8Bob%2Fcrnz7Zx745ziPd6QbPJBASZ6zP%2BGRUE0TOiKnQ6WaHNso%2BK9j5LLqGR0AUIxmruc%2FjTgYxTynZ%2Flv7v0Mcwd9NKz6K02hh7huMzCjGkS4tlkxFzRiYxvahfF7nrVgRnO9er83LWqLycbUohehuw%2Fn2VNEe8UQLv4bPku7yVf2HHLqxrDaFdZUBsg3Gs9N3sPQZQohxM9pcoi8egdMHQ1X4KPD%2B5ZL2WcvTXr%2BnKuycH%2F%2FX%2BHebUxCUs8OkL4rXedWnXkyf4cn2HHmnxQaguUczLQUf8HXfPk%2BKq47i4Pl9dX4ha29YK70X7C1W%2FbJvv68dj3D8fT6SywzP%2FnR394L90N%2BtohPvtX%2BV9j4v6P676j%2Bs0Z1HR%2B5j7IesPDrvfUhreHC%2BXnjv9tfPIpnxvrXyCK6Bgbuv988udn7dljj8wc3UFGBNKXf89vPqWCqcIRZnm0l%2BkjkKuDv7M29rPlvuWtxevS3%2B9WGpXU4xkAfmq4Kdr1cVOdKpo%2FfY63WF3k%2BIU%2F7XCnt9d78Da8mo%2BT%2FIrOJI45Yfb0RNvZCuCkk36032aO%2BAvNI8QlhMBMwtvF6FcMlqb7IPTc79yta%2FzMUjsYo3ItFgwc%2FErmDLtGH9updS0ae0Ki%2Bi6XtORzD%2Br5cXvzAvVL3qrZnENygG%2FQpdkUnHL31HTELlws%2FgpX5N3rAxBR9Lv5EPQdOAVSdhlmJUD%2BShKGGVmMkmIOE2QSa8BgsS23okEeHZZCv5EaXDDvcKnVjFNm23LK07Hi%2FmTe%2BbTd1xgFLrMnleuJIrXdWQ5M1zJtv3sOjSNKM4xDMPox7yU1gtWVocI9BNesVUs3HWR%2FPSUZpFo1rlshZA7fj%2Bs2m%2FiTVZVgjkfusTfxeLM2WZ%2B3X%2FgJEB2Hs3%2FF8W5BPF8%2B5FzBmPYUBYpDznisz8%2BlVg5iERss7XKfB2zDxREeYpXbtqIDvL5GrwUv7p0Fsmmh2ldikMIgo%2BMooYvG578UqqCssnd9N128vX9kbTtGwmzDuQe0VdtCXXg4XkqVxPdcednfyur4Nfe%2FhYlWg7TEYQ7hdX28vHtHnd3yzw3DAzxar4D6I3oywIg0KyHTjAj%2BS4zJv2%2F2SwywpTcB4GwJkgcYhyfwIl%2Be5FOjD2tgVUDTYuVMW%2Fo5U6dZ23bZqvu44TioAEEN6jQBLqHbfxN%2B3Yk85ET53MBX8JBJZHdFogNUZma0gL1iCzzxu64Z%2FvYlLGp2G69%2FWd3lC6nlbYLYQIDl0PN8BncDizvjxNSQR4C%2Fq0M8Nrgtz0vgfg4WKhJ40bDB4xnt9ufVrHENF%2BXQOcrEN4nUb51k2LxrZDXB11pueYy%2Fww0uPte%2BeQKSm31X16UNjm%2FEMGSZLmCRNcWOMNev66VCjqHI6yWHZsAG%2FtF6WeV5pp8nkb3kcRYTfIXbUYVfYJsI%2BlVzIyWS%2Fe77vwd4kRW8EAUXrDmVZoIcYRfB1v8BaYHD7ESdt%2BPDFdc3yKHZvAPOdQSfiGBZ4kHuvND4uFQarMe3txZ2vQ9ukIaaAVhhVl08SW1IC3OSg2fYs9mAVL0Z3Tp8GrWM6hRKArpVDOsG6i9hufYptbAidqciyaY3ZWeiFhOX9vpSJqy7wE4U8UiXDVzq3Dp8KCIaZCx5ue9j50qLx8Mmr%2FegIjJcq068cR89wmU1y37m78MADsh5S2jTPXBs1vaTwCu%2B2TXV%2FrN6SLwTT2hK5nSd8Vp7r0dnRvk7GlN1Yi79NCv%2B4iCHbnNQHrG361gGvHq9vf5rnJJ%2FB3OuH2L1eRlb6%2B0RnqPulk1SKfPpOxj2hFq3uTvqUMnvbkzFU8s6vM9bFVh9%2FWNOa4aqmZY6TsCiKsp2Y1B3SQpNkhnXkoiZkjjdveOpeP8%2FVphRSiKsCT6Gy9wlbF7k%2B2R0ds9ybSKJruOJC0de4%2B5ytuL4KOXIAv4WkoolkDYlrakmxJRqROUlcYAY3EpM1mWwwiH2WIYJ%2BbviTNBgigNltAPefy6xfr1A%2BSbwpoojv74GR7ruILjNgHtQHtbM0jW9xpA9LPnUjm65s4L0E4WDXkecP6s0hZ9Mpd0thw4O3bWqFNgSklkdkCq1hvzPhRXhKLbYD73KGlBGHTxVA%2FD05Hin7fr%2FxFrkpumwddo%2B7PiWfsDethTukT0CvUtOYZvGcamUOyz1YlEVr7VjX7bMqgF%2FIsiwPW3zlVN3ExhwHagxG%2FW5gaE%2FUInFZGuo67eZ4Q0lwRm1Jdfh3U2fm%2BSk9KM8wp2CXorRtxLIsK4U%2FQ50h05RFG33LyBN3I2lixRQaGZU2o0Dv3ju6uNqjxU4dMB0FOuDMstmxSOQivVXEjn6pYnUmmX7Hkib5tL1Rd5ZKPbhdYC5AZcXsOGS8Y45AiXSDbJaOlYVN9KWtRHdvzaJPyN3P1RMCegYhSpJkvsMuEpxtigLtRrE7mcqROyXwhSPms3ABIviBbhfME%2FGuri0aD24eSeJPCVeRTfJy165AtFDkUAFqmSSZm3N5kdL0Kfr6NMUp5nXxedpZTOM0vpKY1%2FuTAD2h07dxrMFOs5r%2Fji7JhiXz%2FcabuuIgDOt6Faw7j3SdNOK4wvEZlhPHR7GtPRdlw5h02Xg8fbhVJrwLC%2FwbkAS5dakobp0PoqxOHdN8L9z3UWep9bxIA6WhVo1sBjJmRe%2FccGCGC%2FR%2FG7R5z6WnsR1mLyNJjrzCRFsKzZ7z5hl%2Fa83AlRQN%2FvQLPDhCeJCeeDeNlV4WOIGVFd1DbaZMiQ6%2FPwcU93ruohUFy352dPcPW5z27KoZHiycDSQm1jqjp1FOcCPxP69e7m8qbKG0k8RtWwLKZiGWT3M9J5kOeZtq6fV6wbMZ4F%2Fu%2BwCHbFqy%2FW9PhkB%2FnDfR7vesb7NlAnEQ%2BXFKx%2FcRFd9ndPxxuMX7zwMvqD%2BOtSh%2BOuwCp%2F84aOP7kI3Xj0f%2FeQ4FJN9fR1H8LxxLgf3%2F9liKgb3962Mp4v4syPrtYE0VShcXaj3NR9GSK2kSLC25m4j4lACERkxbEBQk6pxc0l%2FlA3bBz511E6XweH8aoWlweX48p2mUqYOe2TPP97MT50%2BkBUiIhu1TaahJOQ7dW3DiogZs9xOJ8SuupQxLssd5eQNucOGHe8v5cJqBxhqfIyKeP46IaL6PiBi%2BjojgHnfji2Ic3xTD%2BUExHl8UQzLN%2F63P6B%2Bed10wWuKBByJjedpmZesGG2fGz1EN1p5tQuVNQl1NEbYs3WKe1QYPNUBMdR7Vez2m4droK3thou%2FNWG%2FddiTPV5pcaMC3zJ2kKF4k2c%2FvZTNNksdt9mTuAv0XL%2BpMpXX2srh4h6JumkK4FYUqrOLMfBGEAUCy6DiXvw5MCKeZvyjDZb5IEx4PgOqkZUkqvYCSRG%2Bff5Q3YJHhyjPDvycK24Zn58RhTG0XjRNKFJa%2F0KpNfoomMOSUEBqlV%2FVDDvaKxrM7OebfpKDTg3F2YZWkT7DqMvjlwVwMsZYcwF7YpagYg67RZtmaCJZQfM58aCYAvesizbKbsG5b16469v28o8v9gevXTLx1E80%2FNXpF10%2FzOkVt%2BfQVsP7c014zbdVHwMKP86RHjPraIBf8hvyap%2BeHGzAaeIADjFM0tkAWTbJxniT1lrBEEMfddaKnqrp2HUAHeVrDG1kXZ7BUx1TYB0P3AXe5OwVryNNp53l6y7qz%2Bs5J3iFGWsR48ab0%2FPSMzCysfVpxBik4OB7AIIggyatPl5y1fdqWwGUwK3anWXKYpk1YCRb8j7sL21i40HQw436OpQ7LV6hpwlg9Fc8uZFwftmoL4oV%2FbU8zC644AOfqOlCiD7XCbLBYaqaVqarzbU3EgkULJGQ5lG5G4vGa1QMKCUWSpaR9YRfdhgY%2FbLmPsRo%2BFFRW%2BzG8u5FxdmjpndHQxF2ybRh5sFTqkz2HZV3LBgi12PtztfolS6mos8xg9lZvodhbkdUbQu1w305Ip3YjqVarXjTuFcHqa0XW7WRIZdNITa7aMbRmb90JpcLMDbofZb5B7wLIM1ZNqd089ulYBowOF7prfb8bh9iXYdL%2B3KK4tnRJa1FHSvO8a0wcWS19zTAQrk4cZQaMvW4WPAqiwrH0NtGblcJzL4I2I179%2BpHlcnlcj7obOSFHoQiZ93myZesPJMPUne%2BSjF364bhumi7W2ffpMDSdfXi9RSZrCtERij5o9HPowT%2FgmTw%2BSzlQSecazY4M724TbHdjp0%2ByKP%2FkOwLGfaIRsYzV45S8vmIxzbdomHSAZLTN%2FIF6dg0i1WD2dzRfHuVjX%2Fa7DdgqShQ3CIQWyo8bwEZdd1qgV5nchZ4gRD7CYPksou%2B6TD4t8tV%2B16OsZ2hO6Wmp50D7IXuunCkwhfyUUCoSxidtR3SMjJKnQeUmYj9RXVqc7YsMswUAg16QUbHXRXtvFN6BdS8iOF3WFJwaG7vhnJ7kmjaLBPkP0uwUOuu4c0MYhjDp6WY%2BpnWjm91M8%2BWGZnmGJjW0j3OIvMZEF8SI1Yzi2BRKKfKhuZzCFMIzL3Kc0rctnhpdZuErATedY5hnEd3%2BFYGnpj88wpTQdevhbQCDkc%2FeIOjYPvkBelgoOUS8NNt8GuJ9utl6iukbrcdmk4ksVis%2FXYmTqU3I3Sp84HFKZ34hS7MZ5P0M51LNh9p1lhWq6%2Bf4IEHAWLsfE5Gmp3kjScG%2F3HhscMmXGTKJzdHCwSdUFz271JLNs0xJib%2BNu2jHC0wfWRPlipZu9M4kB0iqMGuLsJXrzVsXrU48e%2FeH97BncYiQA%2FDtzKGhAydxNT%2BXS%2BWmSVeaN1eEabMzTw6Dsrb2c9TCUe34lA4j7T8wyzQa1pebUYlgrK8j57GYk%2FGCHIP1NzAkKg5RX%2Fd6r7FmQR%2Bh6DiLvlp4yfr%2B4ADegEQ24OLbtl1hm417fzi%2BS6FR4YDAyTt9Clyj9no0Be1StwW5OuP6QAuqc7lbinJYiGU%2B61u2vG5mHbw8nR5W2AjGxAOPZWjsxNtmSfiayM4DSTU%2Bg1Tr7asrQiETNKTXwvqdwdYyQngYMwX61t52rJjZzaBYAdlMQi13QF432bTZYoAe5IUQuQ4j2CcO9NBjDFF4b1A6H1DkzO6jaND57MxvjCXJGbLMT3yJebatECKkTpfS6hgamvc53Icbbpt9yS8Clzmf8bEPacO9rq%2Bmy5aNTEGH21Q%2B0gZll0a3xtJ8m4i5L2DhyYN0Jt%2F8ZCSfYZxuhYeS50T1MHfTqNBvkHZ6itLcrjOzkeiWod0dhCPOFvx%2Bz86bv%2BYeAJOUtcAx%2Bt5HoU%2BRTTatxvo96gHpXcQoujfQfURZNLWYSGHnEWUb27882x%2FNbSeGxfG6AyzL1o8ryj7LK2w7ax7brFvRbUEH%2BTt5BjAyynzZDUujdk7cO4g3unDJ49B0FxuCPQFJJDHFy6bjSGeNo5y8r7rf52uq%2B9FDxtdnPjgdH4aN2i2Y5jIfrxWdE8wGadtQ08clE338uRjIhJ9R2WNXyWb8BsM%2BCc4wyswlnGB1m4H0Jk0illZns213r2l%2B0t%2FVcLntTpNJoIcK9w234cATAQS3ZRELrz5cvf54eHHkhgi%2Bt4dHNXES5CDIkttOm8fmA6FhWhq7NhjVhawfhlTsEGtNbXvQQX1%2F9J729B9LjtQ%2BviA10rN1AbCF2MD0i3TWNYzYtBivSYyj8FgcofPMMy9QaeOkT7OzSLdY4Hi5Mdl%2BcHOgf%2BM2ApJdaVGgMCn%2F%2BazcbmTrjaRh4%2Fre4mYKsBmQIjUVZAoT5NJwhI9mcwhZqr28vx2Z1LjUFvErigcAdzXSlnDWEWIoPqJR%2BV5ZAviCilJo7z4%2FtdgDPh5Nq7XKi1SRu1GruG1CrQH273O7kGe4kbYduXn1OeDZhlrChHR3uwBKW8G0tLbAY9NgimK8F68In25ZyMqBty5gyVmvRMl9uqMxQsUk0x0Sw6qCSADgvAzNllu0RDRmZdN%2B48wnBhO%2BwmKcjDlmzK7niO97R26jxotqhFS3sB4K1Ufcu4QMqp6tL%2BVSiyjHl0fzsSUEvaXHTpaKuHqHiQrONmCRDy3Zf9DDWbETuANbI8p41rAqir65cJXwqbI%2Fpi5F15s1p%2Bv%2BtV4s7YHVw760CuOEOQQk5ubIhXj6rdaZI3ri71g%2BpoQVbeaT%2Fgc32uzgH0%2FcJ9atRW8C%2B3nPctONGqxz2vXZwogSYB%2BYac1vQaKqgMrOpoVb07QQweK6fsIfiM%2Bg2BZ0KUz8pnFMJq84pt8mTHTVw%2BDKDWDWN5QixVQ3V0IcHOvJ%2Bkgr3WjWR9Ev5MGhNirSDOL%2BPCd%2F9qChbJ%2FMbkQJt1kPscF0FkSJu2ykariZUd%2F4LctsYRLN2jxZ6ZmxzBX3YYTIFhhyco7bWehDM3yR8seQbiuE2HJk0gSDS6RWmyFcLer23ERW0T8uHA6pYy%2BJ4VxIq%2BlfsbOZappR4T1oUJI67KCCu1gvPrVxgQcEjK3pUBAKqo6FLHgywqsInoexFkxE95fuYB8CtompbOf%2BqI9ZVcJAZbI7IG7GvXsOM%2Ffshsl0SFhFqh8m83jM66oQ7hZhQE4qYIP8S7m7dqDVqCFOxaYSqcbmqrjdtM3g0gnTLYqiEedTuW8YRqrRWGpCrckLoH2XhzO0b5i64V45cv8%2Fwky9tHY9bfOmUBscKn8aRpJKqtbaodZ1XT%2FGy0ql0Qz7Ql7m8YzVFrOsUGkPWHxCWcfcPKJ7MZVLaeYbKn9amTOrpS%2FtVu6P9ogWZ8Rid4JY%2FLDc4%2F2qu%2F2OUfkWz1jQZTQ1juMgQ53qhiH2MHf19e4Y3mSCvf2dBU7jGlxglDxG%2BmK1wwb4R%2FEmchFqpN6dZCLwe1Bn0y5XB4UuskXsCKrarmQ6nkBtiu5PrPHVED2P1FsdAZ6dv7femFxWWltPo0Q1eySWWMCMw1xvT1%2BMS%2B5WDc91Ok58pbTSCBLtvnvAA3dU82n81iULffleX3tyqPsjCoa0imIF%2B9DViDa%2Bd5STI1KtZb2fZu%2FoB7HSS8y9iOTOMzAU5KOLtfGweE9Ypn%2B73R4PKsnh8RpVsoTu8orTg3ASgmXRg9amsDt3ItR8qd6ue3ty1UBrOwLxtfOod3I%2FZv%2Fn2R5vVbJfWJYkQTCFumz2aFHtBKU2VKq0lqyeCb7ONgSezGpLibtasBGGG2KgRb18OH5wbxasDRRPu6%2FMNUVqJqDGBhgq7U0NMoUkSYbL7nnoPPF7sjZyhKljO42kDpshb59I1FBoOsIG%2FCfPNi52%2BRTOZBmLDonvzkmSPRFNvHaugCnUx2Fp2NWcseNhSDfMfhPAH99wlsKp4ZEU60doHnYx%2BAvTSCCYBg%2FVdtC0cC6XG5cB%2FRbaa7oBsgM3mQ3h%2FR%2BWAfpPfsbd1gIYRpdm0bxFWTqB9szLeVL6csUv7%2BscH7Bko%2B18BwXEi1K3grkEIgtbM5XZXgLjBT2ZxhXJsX%2BaPO8wfwIbME%2F5qwHzp7v9f%2BtuOwU2tnY%2BcFCTM%2FoycKO5tNw7m3y2wEFNwCPhkSQBMw4fhUGEj9DzGvVqIIJqeaS2%2BFGxPgM8W4vLFWbAxnjmx0O6HflzbIDbU7PV92VJCvMHb2XJTQYE2HUlmMGAG9PHcgBP5GvdtRYWy4dRaoCPlIPOmouLDIHfM4ZFWy27udj9uoNQDG6azUGKPGHLiDFqpGfZj3w7yMzmTIuBCz9mj%2Boel0ae7G5wH1%2FT9IpMxYp7Fb6LpnQniCaRKPczmns5qH2atTD0kqw4bJC9YirmpgGPRsut8CaUSOU3E6k8PH0CV%2FFUNhTxWh69nHQfnyAo%2BfpppdVRFTPD4H6mayMCThPn4loJ1V4XXUilgFpnJvssrMcfvq5p4gWj3H0O73WZYiQyPvKAMYz6NSA5unm%2F%2Bxnp2dpSXQLfoiaQIUr1kfioFNcfP0AMX83zCtw2Hdb3U5ZR9U28Q4V9C3QTLnGEzL5cGNgyRcOTWu1xnMb3MsJEmow%2BaUfdY6gd8otlixP6Aw%2BRMozdeyTJVBF%2FxUR%2BE0DkdwtRxDHkUy1%2FS0rIuOtLbUtz9XoHegoB14TgjSUsT2%2B%2FAvZYtkp3vcmvlJ1cQchrPKvoS3Z3AcB2jRpVKxcT8qTQhN6%2BnATxVXY0zhfI%2FmN1ptb3lf1qQ8H%2BLHk66a%2BSp%2FnPkqcG%2Byp5iv4sebou3%2BU7Gvpd8nS3tM76lDylf5Q8KXOqwZKnH%2B9%2BrWHwKXni%2Fix5Arb0KXmS%2FyzEIpfyv6P676j%2BU0clH4cscxitvgqCyfSFWhqKhWc%2BXC8vjpJidUQA5FTbEjYwH%2BYxD8%2FIK3rrep7%2Bp%2FVkiKNCE6Pv0vL0qx0JIo1P05JtJ8u%2Bw%2F1Op%2B%2FK5%2BSOmIXFaoexrua6nouquW8BZzMyR6qheXtKTbWe5TfO6O3il%2FFv0WpU122eFbWG2XHNWGEhSA4wK%2FEQeEh7R5Ng1iW2kFVJITSxHFcosum8opBlOmphi3Zh3d4K9Pif1hM8cevuvm36E1KNM%2FVftnuSsFO7xSJ4QIQ1huAFVPzZpdEwJFLuvJD6Lr4mTioW73eslvv6i4f8FA7CoX011fmB55TuW0CnYicwW20rb0UEmKLOJ1xpK2eKixmJhCuzyivtvQNN5F4bzL2CEADd9i9rLaJTLT%2FNJ8E%2FsU7vKuRIVvuRiR1YNNHC0m4Q19mUKrxsHuvmf4IIpeBsFVi6fIOzx%2FTkQZk17fd2wL4LUqIhhbYDmJ3h5%2Fa9vfpZD3oUNmiD5z4bvrNw9ZxFOGTgyJ3WJolkOX6OyN%2Fl3%2BpX%2BTfkp6uxhdb01b6dwioQ%2FVqknT8mHQ4Ii9KtjXb1mmPx48U7q%2FM85XJZptATSjJ53tEldiXfPXIQiz0ppGn5f1rBecizfFsW2EwW4FMaH9OTALyJTD95rylkvXU4QFSKfA4Fz7oBdSuM1haqYj2B%2BNQVo7ni9XUwDKtv%2BK0DoPc5yEkzskAVkb4akK%2FWOevTOrc9S1NYgzGRn%2BMwxvdphvgm%2FFUKjZMIgloyWa7Pz3W%2Fpl9NSdiPljpxoRSufbm%2F2U8FRtvMOE2nnUfGa2cBTvB4huVPXkHNNi4Mx8%2F5Lz4SqSKwqMs72z8FrvRXmx21JWAV%2FCeU%2FWGkDt%2FTX42Yy5%2BNmAAirTV%2BLQugg0L2Cn635O%2FGwn%2FlyT6NmOyKxb5MUVEuw10yuiZUsevq2pc1LcsfdzhWJgB4jYuVXywENkLzKWz5TVbPdWlIwaxI%2FdoFgOy0GRxn3XLT0nLfcTDU7%2B%2FcL54BtnPCgn%2FLA8CFYNHpO4uyFjvDKDcWfYnC%2FGnh5c8fLbwqimEYfmwZy9okybJNh5sUH7fk2sCWbRFIzGqdT%2FtB%2BKMtnD8WrHoIERf8spoAOR%2Fc%2Byf5wgM59vo3rTEu9bZhTwoPab7j7BotZbT4FDzfvxvOaNtCMun5NfXgVfPBb2vme%2FDwgyHGWDwi7mw%2Fpl%2FtOsiPpm7oEeWGnuklG1XzwaO%2FSaqfMxp1pxffzMVPEhl1y8ji3%2BwYHvoBM60%2F6xXPC02FLc9rq2csWRHvRHMlTd%2Bamj1S5HWXf7OCH%2B3aY7tOQ%2BD%2F3%2B1dSbuqSJP%2BNbUHknEpKiKgjAKyYxSZlFHh13cG59yqe8691V8v%2Bnv66e7aKqaQREa8MbwRflOAq2Gj7bO91kDAKWNPw06wrVym8i2ejWHPr023zW97v2GFs7h7Qm%2BKOGZ9i16Cx5tImkglVGmXcqaYR5vfSFZyVpS%2BLqwjtIQIDmN354Kpobg14%2B6xgj7apsm%2FlpvG%2Febe%2FQaRrtN0nXaGSgyjeHMt95WmDVs%2B3lm%2FI7PPRvYILTtVnH458dvNXidXyhX3g3IVHNlv71DTUKuKOlcWO%2F9XzdA8rJRhazqsbCYLlVa8ffkXgwyvKjawzRUCI4RvEpdlT1%2Fy084o8jjfgmxlW3Z2VTJlnbe5fUbHX%2FZsF7AsTYB29jR5m9wuX7UPtB7YKNsvJ2ptGvNdfmD%2FkpGLVS%2FNsugax7FpcYNzA%2F2gtdAi7skrKL1Jv%2Bod9g1EHAWaswdxqsuRwgkU43sS%2BX5b7jzf64ZSmW86D2Tzhj2P%2BTUZTV0vyGMz9bi3z2y30dWr%2Fe0ZlAcTs8Gad%2BUqJ28KEsuERNI0gcbJ3bS%2FrJ2zYbelhirU9fHxODhPotV6j2XnZWm07ub%2BTvax%2Buw%2FiUb6%2BydSeKJXSeS5XR9S2%2BEXbQZ5huUuIW0YkkwdahQ9MTaglutO%2BYYvPt6XweiOs8%2FZIslWIBcONXQIw%2BjCUvLbKzg%2BPbfCfikhiabHqCsuUL9JHrV%2FNjIqQINXo67rM9nswSG3fVKYHGxdfezFKuUSEIwYXer%2FMkrrMZrQLsSg3eUiJs%2BXSrIu7r9YQVLJax9rwt16Q8cgnw90D2pT3hkVsdT02F1Ybuy6mKZZfZfjP5Lyf7XioXqvNMJuZKFo0sTe4%2FV8AAaARD6mEUp7PUmMyv%2FE4j%2BI1KimiEr%2BWuOE8aCtJ02AMOpiJHviFLkSPldBHpTeP6PDu0xtBSxmn%2FIWUZzxWb2%2Fubc8Ow1UaNZMlLqXu1rdhUwbsS9Pc9xlaQTNTcWz56v43seoCY2BWvCbPDXOzP4UW%2FjiQUwJ0LKbUXBDdl%2BNQ8cCkBy8KH89IpscLlXiB%2BTFJb%2FGBX62QiTXq6qlMx907OgIpH6kuYDq2nJNQCTu0UNKlP0dPlXsUh3pOVLDOm%2B57TSaX9DCU1Tb80vt5GZhBuwsHLPvJ6Sytowb3K6ZcP9iH1nT7vOUYF5PUy2%2BejL4RyZ2cfbq%2FLOvBNRusGLK7perrY0klV%2BvfryV%2F9v%2B3j939f%2Firorphg%2FvUHOJCm3F9Ls1q%2BDQOvZzj13b081sw8%2F0UW%2BuM278zqUeaxTuL8xlIVN0ry0ZDuVQL4EwuU7lYivi%2BCSBcTY7YUU0%2BNBc68r19BqvbPf0FzSySS7Yn%2F7ARA226Iet6KfoQYSeV%2BURGtX6aduPVi1nA7BU8Rt9sk8uGPVCa8cgaYnQUVpye8cuOYVdcpbXPTo6eeo%2BkV1yjGOgYMzYIX%2BNW6T8ncfI0RvQ2%2BfDnHRk%2B37nMQeU9AooynEy3syrcX8LKEGO8qJPIq3%2FVtN%2B7HEJuTT7VFg0e6r49GdbkBP3sz5m8ExP4JvTxZU6K3t8T1jbHY4ZkKHMppML65oLBSOixzRlgzi%2FxOCavq0Hl9Qzbxx%2Bo%2F1pLiqbRbBrchd%2FwSgbvcUeAN3bCjfYN6nCz8ey7onHThDFdl1IubnZ%2B3Y1RLXguOSlIhPfv9cAWmfBHvFy3he51dmNqm2psSgQ5YklhgvFqd5JO8B0hygZKE0Ugch%2FPJ7kHQphxM3Z4NzSswWgWf%2FNip2kueTZubrJXf5%2BAraFOKFi%2FsVm%2FG85ef%2Fc1T939T95V7tWgX4UgYKPZaSUW2n31NT5kj%2FCQ%2F3ypVP%2Bdc12zFzpE%2BFHNjX0XTz652p5vpjh1qvB9uv%2FkTBjgWAlV7PpgUNn56KIF1cKr%2FL2S6TEkm%2BJa1iP1%2FXoYhWTAmUhlXdiTy2GtXl81VmP3ugUIGLTfZaghJihObK1Hb%2F6az%2F%2B%2FVCIZdEE1JoseyXHwpuqtaIAkfx0MDLmZQKtGn9gFWBS2FCCp3TwflhBi%2FUeKjR3pGYdf0IQBmTrtfsdqrPL6vf5L4rw4Srl46q%2BPuysoXw9TOXUFK%2BrkSpmR0yQ6lX3PCR4N5ttXbw2x81mfwuAprTZ79Q1SbwRbeuBfQn1Im5eUblkzO19fawBL%2FHUoCxNs%2Bkc8qoIkL1aoGhKgqhZYjfG6As9iIzgmd0bYjJ%2BRD76abrnsiwf9pl%2BihCzIUMhIkJ5gqGVpQ1FP2zZTD1FhywTBJG%2FLjwzWaQ8TSg9WPqdKcksuMpiQgVhcCCDlUAHUQKNpOOtmQHB9%2BkEKfadiBZ6coHH4hw1weHILkuojzIrCVmuT3HJVL3PhrGDkt5L5LDLAe71iC0QT7857jzCF%2F3Tn0ZtJoElIZJOmC9Gkx3MrVrW45xliCQFVOp1d1F3DUlKRlU4WXo%2Fj1C19HZ9f7uHFz5tZd8%2BEFhgYUaDdqwQmqZUcBbuvc5aPB5pnt93GbZS0i7o2Vi1z3LxtNa4wvIkOHqJkufALFxUFMU4kR5a7WDqR%2FnA6IzRNA1KDXmdcxlnaSyk0zqIw8dfTPpgGlJf8BG0fOGhroggpMm3oIiCCRhBcB80wzzzkiQ5w82adJo75sRuiYPRPayFb698qleJ1wE6yh5PIvWMaStCfbYDMc9sUhzE3bMidKVqXKvlU%2B7dU46Czru7zXKuJMg9zyjijlE9339q6GNwjyREeSkX0zLtNqI0Lz7kgadCNhiMeJZM9eB9CYwdLPpHrf46RivZpbcpM3RdIEpZkOW1JVy6pJM7jRALkB%2FeGXUF1bbjWBZZlp2NQTzRf%2FyYezrTTpQ%2BFzRBvOLOeFaknHkqm%2BxChs2RtFGd55nBIl%2Fw0LAx6QffRxSZvhmz0y3uo2crXmjg7CNhtBBpMwATigJ2Xh1lXzQIyUYx3Tt%2FIaEwUIBdmo2XwW5pSEQYDnIFBmsA1ZINtE5%2BxJ9KxTpotXYCqAuWKDWfJm%2BIuHo6eVw8rsx5RfeKPCnXOvV8Lfu7Xy4E8zb4mI7XiQvBa%2FPwWk4RyhNgnKdIJL7HvFEbcuO9KDAeG6A8mcKQiNX28R8%2F%2Bg3gY%2FtGu9trbQqg9wixeCvvEKItXDicpZB0lg2Jp8i%2BZowwHaYsS1Xyj7%2Fo0wn9LqFeXZJI%2BgHvC%2FWBQM1%2Bk87h4LXJct1IezlLycNhlxMtXLGUUAJ77noRi6Eko6xKwnK3vF4bQwFad8SP8ToQsvKpACjUktTOKdvpRLOg%2BZk50zttrVNGoEcruLCTfnO0Gz%2BkxzgZns94ILFQN3Jadn%2BSqGsxhtm1UmMebqFW1p5ABYocj2kv4Y0MDG3sdTrR12yM0xnQPuBQ9dERH0FlCLttKbyePr%2B4IyKwDUDeiYFiKCxCK5E64fnjKUt2a1jMo4Qh8t0m4S6wnEvkb%2Fq9OT4r8GOAsiAWQGa%2FFWitzAVKM%2BfK11ZajAXoM6KbBQ2Njwv0XpC6q9AMVj9Bk4RF9La23FEsi%2FU12gqn3FpE4m546Q2qzsUIchL8aS9D%2BPdCn0A2RJANO%2BSUZFpo3iHGcg%2BzJaWUmolbw%2BT5MJank2Es7%2FQJjKXBon39R%2FtP9REnSVXEQZdaHb%2Bbo8vZCGbgoeX1wnBDVfIsa9sMfTxq6OQJ5raBsJ%2BgsS8TdCe953wLetczT%2Fx0QrQOAfT5gepsI8gGaWaATVMbBGOxLaiuuqaYuQANwHFRkD5XOzOmH7wsa9UHQ4w3jo2Rg5CfIe%2BYzVU2JptwU452FPU8zdoMQ9O3BnlcnWol1H0G9cIpTNyp9y4OVLaR2JOG7icoBZzXaGo9BILRYnOK9ceJQktqFWiJCCq%2FCp2vnc9JSgcRefMgOr3KvTdluY8212eOmMwDngstcEV6787z5MYgvJMvwIENSY6rUd9%2FtgXuqPFVtkplXC6pb0PuyBJc3ZgaCFwd7oE%2BHvbggnW8pEvYXpySMfM9Dzts8E6bGiKI3tp9ghwLdHn4HFgmcbVMZDrspmmpNG5yxMBzq0wrS7BJhygCJRtX5dQb6PpeuNqphf1AtsdBs4nxMLjC2ckljnuJURzzHgGFuMkd7RKS%2BWDekNCSfw7Z7ni5XzYdN3UOIWkOFXrPti8JpllVI1z52da4YVMYUaPzbUW%2B80OkCQ%2FviqUFUa9KDCJBIJHPKdE6XvXGXB9e%2FFMblNAAjdlYoYH2vUQZ%2FOBwvtlvTCgM%2BuQ%2BcVyQQtXXlSylEuOa7SbzOII9S7yZu1VuftCeA%2BuD9qx%2B0p5tsfykPT8%2BaM%2FSn9Tp%2BZM6ffmgTm%2F%2FpE7%2Fe9eA%2FZIX0wwBx2zKjX6E6zb7SnJKezTr7fa%2FiWaNvtKsaf7fyLMGCX48hp%2B%2BO2DEmp8eSQpX%2FAc%3D)