// Where the phone goes for the lung check, in the 200 x 240 chest drawing.
// Front view: the person faces you, so their LEFT is on your right.
// Back view: you see their back, so their LEFT is on your left.

export type Side = 'front' | 'back'

export interface Spot {
  id: string
  name: string
  side: Side
  x: number // position in the 200 x 240 drawing
  y: number
  tip: string
}

export const SPOTS: Spot[] = [
  {
    id: 'front-left',
    name: 'Front left',
    side: 'front',
    x: 130,
    y: 76,
    tip: 'Just below the left collarbone, about a hand-width from the middle of the chest.',
  },
  {
    id: 'front-right',
    name: 'Front right',
    side: 'front',
    x: 70,
    y: 76,
    tip: 'Just below the right collarbone, about a hand-width from the middle of the chest.',
  },
  {
    id: 'back-left',
    name: 'Back left',
    side: 'back',
    x: 72,
    y: 156,
    tip: 'Below the left shoulder blade, about a hand-width from the spine. A helper holds the phone.',
  },
  {
    id: 'back-right',
    name: 'Back right',
    side: 'back',
    x: 128,
    y: 156,
    tip: 'Below the right shoulder blade, about a hand-width from the spine. A helper holds the phone.',
  },
]

/** todo: not done yet; current: do it now; normal / abnormal: its result. */
export type SpotState = 'todo' | 'current' | 'normal' | 'abnormal'
