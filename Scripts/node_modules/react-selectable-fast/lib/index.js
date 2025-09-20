"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var SelectableGroup_1 = require("./SelectableGroup");
exports.SelectableGroup = SelectableGroup_1.SelectableGroup;
var CreateSelectable_1 = require("./CreateSelectable");
exports.createSelectable = CreateSelectable_1.createSelectable;
var SelectAll_1 = require("./SelectAll");
exports.SelectAll = SelectAll_1.SelectAll;
var DeselectAll_1 = require("./DeselectAll");
exports.DeselectAll = DeselectAll_1.DeselectAll;
if (process.env.NODE_ENV === 'development') {
    if (typeof Map !== 'function' ||
        typeof Set !== 'function' ||
        typeof Array.from !== 'function' ||
        typeof Array.isArray !== 'function' ||
        typeof Object.assign !== 'function') {
        throw new Error("\n      React-Selectable-Fast requires Map, Set, Array.from,\n      Array.isArray, and Object.assign to exist.\n      Use a polyfill to provide these for older browsers.\n    ");
    }
}
//# sourceMappingURL=index.js.map