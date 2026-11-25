// tiny-stack-vm: an interpreter for the tiny stack ISA of make.fun's
// "Build a tiny stack virtual machine" bounty. Semantics follow the kit's
// reference VM (generator/subject/vm.mjs); the structure is a parsed opcode
// table plus thrown exit codes: a trap throws, so it can never half-execute
// an instruction, and output already produced is kept. Deterministic.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// Opcode table: byte -> [immediate bytes, pops, pushes]; no row = invalid.
// Row order: HALT PUSH8 PUSH32 POP DUP SWAP OVER ROT ADD SUB MUL DIV MOD
// DIVU MODU NEG AND OR XOR NOT SHL SHR SHRU EQ LT LTU JMP JZ JNZ CALL RET
// LOAD STORE PRINT PUTC IN EXIT.
const DEFS = new Array(256).fill(null);
for (const [op, im, po, pu] of [
  [0,0,0,0],[1,1,0,1],[2,4,0,1],[3,0,1,0],[4,0,1,2],[5,0,2,2],[6,0,2,3],[7,0,3,3],
  [16,0,2,1],[17,0,2,1],[18,0,2,1],[19,0,2,1],[20,0,2,1],[21,0,2,1],[22,0,2,1],[23,0,1,1],
  [24,0,2,1],[25,0,2,1],[26,0,2,1],[27,0,1,1],[28,0,2,1],[29,0,2,1],[30,0,2,1],[32,0,2,1],
  [33,0,2,1],[34,0,2,1],[48,2,0,0],[49,2,1,0],[50,2,1,0],[51,2,0,0],[52,0,0,0],[64,0,1,1],
  [65,0,2,0],[80,0,1,0],[81,0,1,0],[82,0,0,1],[83,0,1,0],
]) DEFS[op] = [im, po, pu];

const STEP_MAX = 1000000;
const OUT_MAX = 65536;

// Runs one instance; returns [stdout bytes, exit code].
function interpret(prog, inp) {
  const mem = new Int32Array(256);
  const S = [];
  const R = [];
  const out = [];
  let pc = 0;
  let steps = 0;
  let ip = 0;
  const fail = (code) => { throw code; };
  try {
    for (;;) {
      // Normative check order: step budget, pc bounds, known opcode,
      // immediate not truncated, stack depth, then the instruction.
      if (steps >= STEP_MAX) fail(10);
      if (pc < 0 || pc >= prog.length) fail(9);
      const op = prog[pc];
      const meta = DEFS[op];
      if (meta === null) fail(5);
      const im = meta[0];
      if (pc + im >= prog.length) fail(9);
      steps += 1;
      if (S.length < meta[1]) fail(2);
      if (S.length - meta[1] + meta[2] > 256) fail(3);
      const arg = im === 2 ? prog[pc + 1] | (prog[pc + 2] << 8) : 0;
      const rel = arg >= 32768 ? arg - 65536 : arg;
      const next = pc + 1 + im;
      pc = next; // jumps below overwrite this default advance
      switch (op) {
        case 0:
          fail(0);
          break;
        case 1: { // PUSH8, sign-extended
          const v = prog[next - 1];
          S.push(v >= 128 ? v - 256 : v);
          break;
        }
        case 2: // PUSH32, little-endian; bitwise assembly is signed already
          S.push(prog[next - 4] | (prog[next - 3] << 8) | (prog[next - 2] << 16) | (prog[next - 1] << 24));
          break;
        case 3:
          S.pop();
          break;
        case 4:
          S.push(S[S.length - 1]);
          break;
        case 5: {
          const t = S[S.length - 1];
          S[S.length - 1] = S[S.length - 2];
          S[S.length - 2] = t;
          break;
        }
        case 6:
          S.push(S[S.length - 2]);
          break;
        case 7: { // ROT: ( a b c -- b c a )
          const c = S.pop();
          const b = S.pop();
          const a = S.pop();
          S.push(b, c, a);
          break;
        }
        case 23: // NEG: | 0 wraps INT_MIN back to itself
          S.push((-S.pop()) | 0);
          break;
        case 27:
          S.push(~S.pop());
          break;
        case 48:
          pc = next + rel;
          break;
        case 49:
          if (S.pop() === 0) pc = next + rel;
          break;
        case 50:
          if (S.pop() !== 0) pc = next + rel;
          break;
        case 51:
          if (R.length >= 64) fail(7);
          R.push(next);
          pc = arg;
          break;
        case 52:
          if (R.length === 0) fail(8);
          pc = R.pop();
          break;
        case 64: {
          const addr = S[S.length - 1];
          if (addr < 0 || addr > 255) fail(6);
          S[S.length - 1] = mem[addr];
          break;
        }
        case 65: { // STORE: ( v addr -- ), address on top
          const addr = S.pop();
          if (addr < 0 || addr > 255) fail(6);
          mem[addr] = S.pop();
          break;
        }
        case 80: { // PRINT: the whole string fits, or none of it is written
          const text = String(S[S.length - 1]);
          if (out.length + text.length > OUT_MAX) fail(11);
          S.pop();
          for (let i = 0; i < text.length; i++) out.push(text.charCodeAt(i));
          break;
        }
        case 81:
          if (out.length + 1 > OUT_MAX) fail(11);
          out.push(S.pop() & 255);
          break;
        case 82: // IN: -1 once the input is exhausted
          S.push(ip < inp.length ? inp[ip++] : -1);
          break;
        case 83: // EXIT
          fail(S.pop() & 255);
          break;
        default: { // binary operators: ( a b -- r ), b on top
          const b = S.pop();
          const a = S.pop();
          let r;
          if (op === 16) r = (a + b) | 0;
          else if (op === 17) r = (a - b) | 0;
          else if (op === 18) r = Math.imul(a, b); // exact low 32 bits
          else if (op === 19) { // DIV, truncating; INT_MIN / -1 wraps to INT_MIN
            if (b === 0) fail(4);
            r = a === -2147483648 && b === -1 ? a : Math.trunc(a / b) | 0;
          } else if (op === 20) { // MOD, sign follows the dividend
            if (b === 0) fail(4);
            r = b === -1 ? 0 : (a % b) | 0;
          } else if (op === 21) { // DIVU, operands read unsigned
            if (b === 0) fail(4);
            r = Math.floor((a >>> 0) / (b >>> 0)) | 0;
          } else if (op === 22) { // MODU
            if (b === 0) fail(4);
            r = ((a >>> 0) % (b >>> 0)) | 0;
          } else if (op === 24) r = a & b;
          else if (op === 25) r = a | b;
          else if (op === 26) r = a ^ b;
          else if (op === 28) r = a << (b & 31); // shift counts are masked to 5 bits
          else if (op === 29) r = a >> (b & 31);
          else if (op === 30) r = (a >>> (b & 31)) | 0;
          else if (op === 32) r = a === b ? 1 : 0;
          else if (op === 33) r = a < b ? 1 : 0;
          else r = (a >>> 0) < (b >>> 0) ? 1 : 0; // LTU, unsigned
          S.push(r);
          break;
        }
      }
    }
  } catch (code) {
    if (typeof code !== 'number') throw code;
    return [Uint8Array.from(out), code];
  }
}

// Driver: one pass over every instance directory, sorted.
const inDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of readdirSync(inDir).sort()) {
  const prog = new Uint8Array(readFileSync(join(inDir, id, 'program.bin')));
  const inp = new Uint8Array(readFileSync(join(inDir, id, 'input.bin')));
  const [bytes, code] = interpret(prog, inp);
  mkdirSync(join(outDir, id), { recursive: true });
  writeFileSync(join(outDir, id, 'stdout.bin'), bytes);
  writeFileSync(join(outDir, id, 'exit.txt'), `${code}\n`);
}
