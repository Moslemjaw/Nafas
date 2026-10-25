import { useState } from 'react'
import CoughView from './CoughView'
import LungView from './LungView'
import './App.css'

type Tab = 'cough' | 'lung'

export default function App() {
  const [tab, setTab] = useState<Tab>('cough')
  // the lung screen mounts on first visit (it downloads a 45 MB model), then stays
  const [lungOpened, setLungOpened] = useState(false)

  function open(t: Tab) {
    setTab(t)
    if (t === 'lung') setLungOpened(true)
  }

  return (
    <main>
      <header>
        <h1>Nafas</h1>
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
