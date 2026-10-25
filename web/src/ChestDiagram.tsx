// Front / back torso drawings with the lungs and the spots where the phone goes.
import type { Side, Spot, SpotState } from './spots'

const TORSO =
  'M86 10 L86 30 C70 35 42 37 29 51 C21 60 23 92 29 122 L35 230 L165 230 L171 122 ' +
  'C177 92 179 60 171 51 C158 37 130 35 114 30 L114 10 Z'
const LUNG_VIEWER_LEFT = 'M93 50 C71 47 52 68 47 108 C43 148 47 186 56 198 C70 203 86 194 93 180 Z'
const LUNG_VIEWER_RIGHT = 'M107 50 C129 47 148 68 153 108 C157 148 153 186 144 198 C130 203 114 194 107 180 Z'

export function ChestDiagram({
  side,
  spots,
  caption = true,
}: {
  side: Side
  spots: { spot: Spot; state: SpotState; n?: number }[]
  caption?: boolean
}) {
  const leftLabel = side === 'front' ? 'R' : 'L' // the person's side, on the viewer's left
  const rightLabel = side === 'front' ? 'L' : 'R'
  return (
    <figure className="chest">
      <svg viewBox="0 0 200 240" role="img" aria-label={`${side} of the chest with phone positions`}>
        <path d={TORSO} className="torso" />
        <path d={LUNG_VIEWER_LEFT} className="lung" />
        <path d={LUNG_VIEWER_RIGHT} className="lung" />
        {side === 'front' ? (
          <>
            <path d="M56 46 Q78 54 96 42" className="bone" />
            <path d="M144 46 Q122 54 104 42" className="bone" />
            <line x1="100" y1="44" x2="100" y2="150" className="bone thin" />
          </>
        ) : (
          <>
            <line x1="100" y1="30" x2="100" y2="226" className="bone thin dashed" />
            <path d="M56 62 L92 70 L82 130 Z" className="blade" />
            <path d="M144 62 L108 70 L118 130 Z" className="blade" />
          </>
        )}
        <text x="14" y="24" className="side-label">{leftLabel}</text>
        <text x="186" y="24" className="side-label" textAnchor="end">{rightLabel}</text>
        {spots.map(({ spot, state, n }) => (
          <g key={spot.id} className={`spot ${state}`}>
            {state === 'current' && <circle cx={spot.x} cy={spot.y} r="16" className="pulse" />}
            <circle cx={spot.x} cy={spot.y} r="11" />
            {n !== undefined && (
              <text x={spot.x} y={spot.y + 4} textAnchor="middle">
                {n}
              </text>
            )}
          </g>
        ))}
      </svg>
      {caption && <figcaption>{side === 'front' ? 'Front (facing you)' : 'Back'}</figcaption>}
    </figure>
  )
}
