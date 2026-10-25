import { lazy, Suspense, useEffect, useState } from 'react'
import CoughView from './CoughView'
import LungView from './LungView'
import { feed, useFeedStatus } from './feed'
import { forgetPairing, pairingFromUrl } from './link'
import './App.css'

// the big-screen view (and its QR code library) loads only on a computer that asks for it
const DisplayView = lazy(() => import('./DisplayView'))

type Tab = 'cough' | 'lung'

/** #display on a computer turns the app into the big-screen view for a paired phone. */
const isDisplay = () => location.hash.startsWith('#display')

/** Opened from a screen's QR code (#pair=...): mirror this phone to that screen. */
const initialPairing = isDisplay() ? null : pairingFromUrl()

export default function App() {
  const [display, setDisplay] = useState(isDisplay)

  useEffect(() => {
    const onHash = () => setDisplay(isDisplay())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  return display ? (
    <Suspense fallback={null}>
      <DisplayView />
    </Suspense>
  ) : (
    <PhoneApp />
  )
}

function PhoneApp() {
  const [tab, setTab] = useState<Tab>('cough')
  // the lung screen mounts on first visit (it downloads a 45 MB model), then stays
  const [lungOpened, setLungOpened] = useState(false)
  const [paired, setPaired] = useState(initialPairing !== null)
  const status = useFeedStatus()

  useEffect(() => {
    if (initialPairing) feed.connect(initialPairing)
  }, [])

  function open(t: Tab) {
    setTab(t)
    if (t === 'lung') setLungOpened(true)
    feed.update({ tab: t })
  }

  function unpair() {
    feed.disconnect()
    forgetPairing()
    setPaired(false)
  }

  return (
    <main>
      <header>
        <div className="row">
          <h1>Nafas</h1>
          {paired && (
            <span className={`pair-badge ${status}`}>
              <span className="pair-dot" />
              {status === 'connected' ? 'On screen' : 'Connecting…'}
              <button className="link" onClick={unpair}>
                Disconnect
              </button>
            </span>
          )}
        </div>
        <nav className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === 'cough'} onClick={() => open('cough')}>
            Coughs
          </button>
          <button role="tab" aria-selected={tab === 'lung'} onClick={() => open('lung')}>
            Lung check
          </button>
        </nav>
      </header>
      {/* both stay mounted so switching tabs keeps their state */}
      <div className="view" hidden={tab !== 'cough'}>
        <CoughView />
      </div>
      {lungOpened && (
        <div className="view" hidden={tab !== 'lung'}>
          <LungView />
        </div>
      )}
    </main>
  )
}
