traefik_lrt_manifest = 'load-balancers/traefik-lrt'
traefik_lrt_function = 'traefik-lrt'
traefik_rr_manifest = 'load-balancers/traefik-rr'
traefik_rr_function = 'traefik-rr'

# Image size values taken from: https://hub.docker.com/_/traefik?tab=tags&page=1&ordering=last_updated
# parametrization: (image, size, arch)
all_lb_images = [
    (traefik_lrt_manifest, '27M', 'x86'),
    (traefik_lrt_manifest, '27M', 'amd64'),
    (traefik_lrt_manifest, '25M', 'arm32v7'),
    (traefik_lrt_manifest, '25M', 'arm32'),
    (traefik_lrt_manifest, '25M', 'arm'),
    (traefik_lrt_manifest, '25M', 'aarch64'),
    (traefik_lrt_manifest, '25M', 'arm64'),
    (traefik_rr_manifest, '27M', 'x86'),
    (traefik_rr_manifest, '27M', 'amd64'),
    (traefik_rr_manifest, '25M', 'arm32v7'),
    (traefik_rr_manifest, '25M', 'arm32'),
    (traefik_rr_manifest, '25M', 'arm'),
    (traefik_rr_manifest, '25M', 'aarch64'),
    (traefik_rr_manifest, '25M', 'arm64'),
]

