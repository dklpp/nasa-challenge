"use client";

export function Toast({ message }: { message: string }) {
  return (
    <div id="toast" style={{ display: message ? "block" : "none" }}>
      {message}
    </div>
  );
}

export default Toast;
