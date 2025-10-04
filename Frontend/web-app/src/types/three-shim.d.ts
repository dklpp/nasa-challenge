declare module 'three' {
  const Three: any;
  export = Three;
}

declare module 'three/examples/jsm/controls/OrbitControls.js' {
  export class OrbitControls {
    constructor(object: any, domElement?: HTMLElement);
    enableDamping: boolean;
    dampingFactor: number;
    minDistance: number;
    maxDistance: number;
    target: { x: number; y: number; z: number; set: (x: number, y: number, z: number) => void; lerp: (v: any, t: number) => void };
    update(): void;
  }
}
