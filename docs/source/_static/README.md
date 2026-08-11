# Branding assets

`conf.py` picks these up automatically — drop a file in, rebuild, done. No config
edit needed. SVG is preferred over PNG; both are recognised.

| File | Used for |
|---|---|
| `logo.svg` / `logo.png` | Sidebar logo in both themes, and the favicon if no `favicon.*` exists |
| `logo-light.svg` | Sidebar logo in light mode only (overrides `logo.*`) |
| `logo-dark.svg` | Sidebar logo in dark mode only (overrides `logo.*`) |
| `favicon.svg` / `favicon.png` | Browser tab icon (overrides `logo.*`) |

## Contrast

Furo's sidebar is near-white in light mode and near-black in dark mode, so a single
logo has to survive both. The MICM cube is navy, red and light grey: the light-grey
faces nearly vanish on a white sidebar, and the navy nearly vanishes on a black one.
If that bothers you, supply `logo-light.svg` and `logo-dark.svg` instead of one
`logo.svg` — same artwork, with the low-contrast faces lightened or darkened.

A transparent background is worth having either way; a baked-in white box shows as a
white rectangle in dark mode.
