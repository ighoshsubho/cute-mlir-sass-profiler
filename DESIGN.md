---
version: alpha
name: CuTe Compiler Explorer
colors:
  primary: "#9ADBA9"
  on-primary: "#13221A"
  surface: "#191E1E"
  on-surface: "#E2EAE8"
  warning: "#E8C277"
typography:
  body-md:
    fontFamily: Segoe UI, system-ui, sans-serif
    fontSize: 14px
  code:
    fontFamily: SF Mono, Cascadia Code, Liberation Mono, monospace
    fontSize: 12px
rounded:
  sm: 5px
spacing:
  sm: 8px
  md: 12px
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    rounded: "{rounded.sm}"
---

## Overview

Code-first inspector. The code is the workspace; chrome is for opening, navigating, and comparing artifacts.

## Colors

Green means a source link. Amber identifies approximate path associations or heuristic findings. Do not encode certainty in color alone: pair it with the status legend or a confidence label. Light mode is defined alongside the dark palette in `src/style.css`.

## Typography

Use the monospace stack for files, line numbers, code, and evidence. Use the UI stack for actions and explanations.

## Layout

A resizable 2×2 grid with a single optional sidebar. Stack panes on narrow screens without overflowing the page. Keep panes independently scrollable.

## Do's and Don'ts

Use read-only editors, label missing artifacts and metadata, and keep the controls next to the code they operate on. Avoid charts, fictitious timing data, and visual decoration that consumes editor space.
