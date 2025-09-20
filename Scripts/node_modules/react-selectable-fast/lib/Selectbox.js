"use strict";
var __read = (this && this.__read) || function (o, n) {
    var m = typeof Symbol === "function" && o[Symbol.iterator];
    if (!m) return o;
    var i = m.call(o), r, ar = [], e;
    try {
        while ((n === void 0 || n-- > 0) && !(r = i.next()).done) ar.push(r.value);
    }
    catch (error) { e = { error: error }; }
    finally {
        try {
            if (r && !r.done && (m = i["return"])) m.call(i);
        }
        finally { if (e) throw e.error; }
    }
    return ar;
};
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (Object.hasOwnProperty.call(mod, k)) result[k] = mod[k];
    result["default"] = mod;
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
var react_1 = __importStar(require("react"));
var initialState = {
    y: 0,
    x: 0,
    width: 0,
    height: 0,
};
function Selectbox(props) {
    var fixedPosition = props.fixedPosition, getSetState = props.getSetState, className = props.className;
    var _a = __read(react_1.useState(initialState), 2), state = _a[0], setState = _a[1];
    react_1.useEffect(function () {
        getSetState(setState);
    }, []);
    var boxStyle = {
        left: state.x,
        top: state.y,
        width: state.width,
        height: state.height,
        zIndex: 9000,
        position: fixedPosition ? 'fixed' : 'absolute',
        cursor: 'default',
        willChange: 'transform',
        transform: 'translateZ(0)',
    };
    return react_1.default.createElement("div", { className: className, style: boxStyle });
}
exports.Selectbox = Selectbox;
Selectbox.defaultProps = {
    className: 'selectable-selectbox',
};
//# sourceMappingURL=Selectbox.js.map