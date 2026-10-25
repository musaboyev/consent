import React from 'react';
import {createRoot} from 'react-dom/client';
import App from './App';
import './style.css';
class DashboardBoundary extends React.Component<{children:React.ReactNode},{error:string|null}>{
 state:{error:string|null}={error:null};
 static getDerivedStateFromError(error:Error){return {error:error.message}}
 render(){return this.state.error?<div className="shell"><div className="alert" role="alert">Dashboard could not render: {this.state.error}. Reload this page. The local camera pipeline continues independently.</div></div>:this.props.children}
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><DashboardBoundary><App/></DashboardBoundary></React.StrictMode>);
