# Training Data Storage

## Current layout

As of 2026-10-09, training datasets are stored on the dedicated `/data` filesystem instead of the root filesystem.

- Physical location: `/data/zdog-yizhu/server-cache/training`
- Repository-compatible path: `server/.cache/training`
- Mount method: persistent bind mount
- `/etc/fstab` entry:

```fstab
/data/zdog-yizhu/server-cache/training /home/zhang/workspace/zdog-yizhu/server/.cache/training none bind,nofail,x-systemd.requires-mounts-for=/data 0 0
```

A bind mount is used instead of a symbolic link because training code validates that resolved image paths remain under the repository root. Existing manifests and commands therefore continue to use `server/.cache/training/...` without modification.

Only the training cache was moved. Model weights, extracted feature artifacts, catalog databases, private application photos, and source code remain in their existing locations.

## Verification

Check both the data filesystem and the bind mount before starting collection or training:

```sh
df -h / /data
findmnt --target /home/zhang/workspace/zdog-yizhu/server/.cache/training
test -r server/.cache/training/china-common-1000/species.json
```

Expected source in `findmnt`:

```text
/dev/sda6[/yizhu/server-cache/training]
```

Immediately after migration, 159,643 files totaling 11,202,748,370 bytes were compared between source and destination with an `rsync` dry run before the old root-filesystem copy was removed. Root usage decreased from 87% to 73%; `/data` usage became 57%.

## Recovery and maintenance

`/data` must be mounted before the bind mount. The `x-systemd.requires-mounts-for=/data` option records this dependency. After changing `/etc/fstab`, run:

```sh
sudo systemctl daemon-reload
sudo findmnt --verify --verbose
sudo mount -a
```

Do not replace the bind mount with a symlink, and do not write a second dataset copy under the mount point while `/data` is unavailable. If the mount is missing, stop training or collection, restore `/data`, then run `sudo mount -a` and verify with `findmnt`.
